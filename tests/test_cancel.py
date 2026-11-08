"""Stopping a scan leaves every file either fully processed or untouched."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import FakeRecognizer, person
from recognize import web
from recognize.match import PersonHit, ScanOptions
from recognize.scan import InspectionLike, ScanCancelled, ScanHooks, run_scan


def _four(tmp_path: Path) -> Path:
    source = tmp_path / "in"
    source.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        (source / name).write_bytes(name.encode())
    return source


def test_run_scan_stops_between_files_and_writes_a_partial_report(tmp_path: Path) -> None:
    source = _four(tmp_path)
    hit = PersonHit("emre", "Emre", 0.8, 1, False)
    fake = FakeRecognizer({name: [hit] for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg")})
    dest = tmp_path / "out"
    report = run_scan(
        source,
        dest,
        fake,
        [person("Emre", "emre")],
        ScanOptions(move=True),
        should_stop=lambda: len(fake.seen) >= 2,
    )
    assert report["complete"] is False
    assert [row["source"] for row in report["results"]] == ["a.jpg", "b.jpg"]
    assert report["progress"] == {"done": 2, "total": 4}
    assert report["stoppedAt"]
    assert sorted(p.name for p in source.iterdir()) == ["c.jpg", "d.jpg"]
    assert sorted(p.name for p in (dest / "Emre").iterdir()) == ["a.jpg", "b.jpg"]
    saved = json.loads((dest / "rapor.json").read_text(encoding="utf-8"))
    assert saved["progress"] == {"done": 2, "total": 4} and saved["complete"] is False
    assert saved["summary"]["files"] == 2


def test_stop_inside_a_video_is_not_a_file_error(tmp_path: Path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    (source / "a.jpg").write_bytes(b"a")
    (source / "uzun.mp4").write_bytes(b"video")
    flag = {"stop": False}
    frames: list[str] = []

    class _Video(FakeRecognizer):
        def inspect_video(self, path: Path, progress=None) -> InspectionLike:
            for index in range(5):
                if index == 2:
                    flag["stop"] = True
                frames.append(f"{index}")
                if progress is not None:
                    progress(f"{path.name} · {index + 1}. kare")
            raise AssertionError("the stop must interrupt the frames")

    hit = PersonHit("emre", "Emre", 0.8, 1, False)
    dest = tmp_path / "out"
    report = run_scan(
        source,
        dest,
        _Video({"a.jpg": [hit]}),
        [person("Emre", "emre")],
        ScanOptions(),
        should_stop=lambda: flag["stop"],
    )
    assert frames == ["0", "1", "2"]
    assert [row["source"] for row in report["results"]] == ["a.jpg"]
    assert report["summary"]["errors"] == 0
    assert (source / "uzun.mp4").is_file()
    assert [p.name for p in (dest / "Emre").iterdir()] == ["a.jpg"]


def test_stop_before_any_file_keeps_the_previous_report(tmp_path: Path) -> None:
    source = _four(tmp_path)
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "rapor.json").write_text('{"eski": true}', encoding="utf-8")
    archived: list[Path] = []
    with pytest.raises(ScanCancelled):
        run_scan(
            source,
            dest,
            FakeRecognizer({}),
            [person("Emre", "emre")],
            ScanOptions(),
            should_stop=lambda: True,
            hooks=ScanHooks(archive=archived.append),
        )
    assert (dest / "rapor.json").read_text(encoding="utf-8") == '{"eski": true}'
    assert archived == []


def test_cancel_endpoint(home) -> None:
    with TestClient(web.app) as client:
        idle = client.post("/api/tara/durdur")
        assert idle.status_code == 409
        assert idle.json()["detail"] == "Süren bir tarama yok."
        with web._scan_lock:
            web._scan["state"] = "running"
        try:
            first = client.post("/api/tara/durdur")
            again = client.post("/api/tara/durdur")
            status = client.get("/api/tara").json()
        finally:
            with web._scan_lock:
                web._scan["state"] = "idle"
                web._scan["cancelRequested"] = False
            web._cancel.clear()
    assert first.json() == {"state": "cancelling"} and again.json() == {"state": "cancelling"}
    assert status["state"] == "running" and status["cancelRequested"] is True
    assert status["current"] == "Durduruluyor…"
