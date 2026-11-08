"""Opening folders never leaves the output, and a test never opens a real window."""

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import scanned_output
from recognize import reveal, web
from recognize.errors import AppError
from recognize.web import app


class _Runs:
    def __init__(self, *codes: int) -> None:
        self.codes = list(codes)
        self.calls: list[list[str]] = []

    def __call__(self, command, **kwargs) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(command))
        return subprocess.CompletedProcess(command, self.codes.pop(0), "", "")


def test_mac_opens_and_reveals_with_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _Runs(0, 0, 1)
    monkeypatch.setattr(reveal.sys, "platform", "darwin")
    monkeypatch.setattr(reveal.subprocess, "run", runs)
    reveal.open_folder(tmp_path)
    reveal.reveal_file(tmp_path / "a.jpg")
    assert runs.calls == [["open", str(tmp_path)], ["open", "-R", str(tmp_path / "a.jpg")]]
    with pytest.raises(AppError, match="Finder"):
        reveal.open_folder(tmp_path)


def test_windows_uses_startfile_and_explorer_select(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    popened: list[list[str]] = []
    monkeypatch.setattr(reveal.sys, "platform", "win32")
    monkeypatch.setattr(reveal.os, "startfile", lambda path: opened.append(path), raising=False)

    class _Popen:
        def __init__(self, command, **kwargs) -> None:
            popened.append(list(command))

    monkeypatch.setattr(reveal.subprocess, "Popen", _Popen)
    reveal.open_folder(tmp_path)
    reveal.reveal_file(tmp_path / "a.jpg")
    assert opened == [str(tmp_path)]
    assert popened == [["explorer", "/select,", str(tmp_path / "a.jpg")]]
    assert "Gezgin" in reveal.failure_message()


def test_linux_needs_xdg_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reveal.sys, "platform", "linux")
    monkeypatch.setattr(reveal.shutil, "which", lambda name: None)
    with pytest.raises(AppError, match="açın"):
        reveal.open_folder(tmp_path)


def test_endpoint_opens_only_report_folders(
    home, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = scanned_output(tmp_path)
    opened: list[Path] = []
    revealed: list[Path] = []
    monkeypatch.setattr(web, "open_folder", lambda path: opened.append(path))
    monkeypatch.setattr(web, "reveal_file", lambda path: revealed.append(path))
    monkeypatch.setattr(web, "_last_open", ("", 0.0))
    with TestClient(app) as client:
        base = {"output": str(output)}
        ok = client.post("/api/klasoru-ac", json={**base, "slug": "emre"})
        assert ok.status_code == 200, ok.text
        assert ok.json() == {
            "opened": str(output / "Emre Kaya"),
            "selected": None,
            "already": False,
        }
        root = client.post("/api/klasoru-ac", json=base)
        assert root.json()["opened"] == str(output)
        file = client.post("/api/klasoru-ac", json={**base, "path": "Emre Kaya/emre.jpg"})
        assert file.json()["selected"] == str(output / "Emre Kaya" / "emre.jpg")
        unknown = client.post("/api/klasoru-ac", json={**base, "slug": "kimse"})
        assert (unknown.status_code, unknown.json()["detail"]) == (404, "Bu kişi bu taramada yok.")
        escape = client.post("/api/klasoru-ac", json={**base, "path": "../etkinlik"})
        assert escape.status_code == 404
        two = client.post("/api/klasoru-ac", json={**base, "slug": "emre", "path": "x"})
        assert (two.status_code, two.json()["detail"]) == (
            400,
            "Açılacak yeri tek bir şekilde belirtin.",
        )
        foreign = client.post(
            "/api/klasoru-ac", json=base, headers={"Origin": "https://example.com"}
        )
        assert foreign.status_code == 403
        no_rejects = client.post("/api/klasoru-ac", json={**base, "target": "rejected"})
        assert no_rejects.status_code == 404
    assert opened == [output / "Emre Kaya", output]
    assert revealed == [output / "Emre Kaya" / "emre.jpg"]


def test_double_click_does_not_open_twice(
    home, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = scanned_output(tmp_path)
    opened: list[Path] = []
    monkeypatch.setattr(web, "open_folder", lambda path: opened.append(path))
    monkeypatch.setattr(web, "_last_open", ("", 0.0))
    with TestClient(app) as client:
        first = client.post("/api/klasoru-ac", json={"output": str(output), "slug": "ali"})
        second = client.post("/api/klasoru-ac", json={"output": str(output), "slug": "ali"})
    assert first.json()["already"] is False and second.json()["already"] is True
    assert len(opened) == 1


def test_launcher_failure_returns_the_path_to_copy(
    home, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = scanned_output(tmp_path)

    def fail(path: Path) -> None:
        raise AppError(reveal.failure_message())

    monkeypatch.setattr(web, "open_folder", fail)
    monkeypatch.setattr(web, "_last_open", ("", 0.0))
    with TestClient(app) as client:
        response = client.post("/api/klasoru-ac", json={"output": str(output)})
    assert response.status_code == 500
    assert response.json()["path"] == str(output)
    assert response.json()["detail"].startswith("Klasör açılamadı.")
