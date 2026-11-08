"""Folder picker: Finder on macOS, Explorer on Windows, typed path as the fallback."""

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from recognize import picker
from recognize.errors import AppError


class _Runs:
    def __init__(self, *results: subprocess.CompletedProcess[str]) -> None:
        self.results = list(results)
        self.calls: list[list[str]] = []

    def __call__(self, command, **kwargs) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(command))
        return self.results.pop(0)


def _done(code: int, out: str = "", err: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], code, out, err)


def test_mac_uses_finder_and_starts_in_the_nearest_existing_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chosen = tmp_path / "Etkinlik"
    chosen.mkdir()
    runs = _Runs(_done(0, f"{chosen}/\n"))
    monkeypatch.setattr(picker.sys, "platform", "darwin")
    monkeypatch.setattr(picker.subprocess, "run", runs)
    got = picker.pick_folder("Klasör seçin", str(tmp_path / "yok" / "daha-da-yok"))
    assert got == str(chosen.resolve())
    command = runs.calls[0]
    assert command[0] == "osascript"
    assert command[-2:] == ["Klasör seçin", str(tmp_path)]


def test_mac_cancel_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = _Runs(_done(1, err="execution error: User canceled. (-128)"))
    monkeypatch.setattr(picker.sys, "platform", "darwin")
    monkeypatch.setattr(picker.subprocess, "run", runs)
    assert picker.pick_folder("Klasör seçin") is None


def test_mac_failure_asks_for_a_typed_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(picker.sys, "platform", "darwin")
    monkeypatch.setattr(picker.subprocess, "run", _Runs(_done(1, err="no display")))
    with pytest.raises(AppError, match="elle"):
        picker.pick_folder("Klasör seçin")


def test_windows_falls_back_to_powershell_when_tk_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = _Runs(_done(1, err="No module named tkinter"), _done(0, str(tmp_path)))
    monkeypatch.setattr(picker.sys, "platform", "win32")
    monkeypatch.setattr(picker.subprocess, "run", runs)
    monkeypatch.setattr(picker.shutil, "which", lambda name: "powershell.exe")
    assert picker.pick_folder("Klasör seçin", str(tmp_path)) == str(tmp_path.resolve())
    assert runs.calls[0][1] == "-c"
    assert runs.calls[1][0] == "powershell.exe"
    assert "-STA" in runs.calls[1]


def test_endpoint_returns_the_choice_and_refuses_other_sites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recognize import web

    seen: list[tuple[str, str]] = []

    def fake(title: str, current: str) -> str:
        seen.append((title, current))
        return str(tmp_path)

    monkeypatch.setattr(web, "pick_folder", fake)
    with TestClient(web.app) as client:
        ok = client.post("/api/klasor-sec", json={"purpose": "destination", "current": "/x"})
        bad = client.post(
            "/api/klasor-sec",
            json={"purpose": "source"},
            headers={"Origin": "https://example.com"},
        )
        local = client.post(
            "/api/klasor-sec",
            json={"purpose": "source"},
            headers={"Origin": "http://127.0.0.1:8765"},
        )
    assert ok.status_code == 200
    assert ok.json() == {"path": str(tmp_path)}
    assert seen[0] == ("Kişi klasörlerinin yazılacağı klasörü seçin", "/x")
    assert bad.status_code == 403
    assert local.status_code == 200
    assert len(seen) == 2
