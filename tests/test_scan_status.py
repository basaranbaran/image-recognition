"""The richer scan status: context, per-person counts, the last match and the estimate."""

import time
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from recognize import web
from recognize.eta import Plan, estimate_eta, plan_units
from recognize.gallery import add_references


def test_eta_is_null_until_enough_work_is_done() -> None:
    assert estimate_eta(1.0, 10.0, 2.0, 1) is None
    assert estimate_eta(3.0, 10.0, 3.0, 3) == 10
    assert estimate_eta(1.0, 10.0, 6.0, 1) == 55
    assert estimate_eta(0.0, 10.0, 6.0, 0) is None
    assert estimate_eta(10.0, 10.0, 6.0, 10) == 0


def test_eta_weights_video_samples(tmp_path: Path) -> None:
    files = [tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "klip.mp4"]
    plan = plan_units(files, 0.5, duration_of=lambda path: 4.0)
    assert plan.photos == 2 and plan.frames[files[2]] == 8
    assert plan.total_units == 2 + 8 * 0.5
    plan.finish_file(files[0], 1.0)
    plan.finish_file(files[1], 1.0)
    assert plan.done_units == 2.0
    plan.start_file(files[2])
    for _ in range(3):
        plan.frame()
    assert plan.done_units == 2 + 3 * 0.5
    # The video took two seconds per frame while a photo took one: frames now weigh 2.
    plan.finish_file(files[2], 16.0)
    assert plan.weight == 2.0
    assert plan.done_units == plan.total_units == 2 + 8 * 2.0
    assert Plan().total_units == 0


def _enroll(home: Path, name: str) -> str:
    person, _created = add_references(
        name,
        [(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 0.9, b"jpeg", "on.jpg")],
        None,
        update_clothing=False,
    )
    return person.slug


def test_status_carries_the_scan_context(home, tmp_path: Path, monkeypatch) -> None:
    slug = _enroll(home, "Emre Kaya")
    source = tmp_path / "etkinlik"
    source.mkdir()
    (source / "a.jpg").write_bytes(b"a")
    (source / "b.jpg").write_bytes(b"b")
    dest = tmp_path / "cikti"
    monkeypatch.setattr(web, "_get_engine", lambda: object())
    monkeypatch.setattr(web, "GalleryRecognizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(web, "_models_missing", lambda: False)

    def fake_run(src, dst, recognizer, people, options, progress, *, should_stop, hooks):
        hooks.on_plan([src / "a.jpg", src / "b.jpg"])
        hooks.on_phase("files")
        progress(0, 2, "a.jpg")
        record = {
            "source": "a.jpg",
            "kind": "image",
            "matches": [{"name": "Emre Kaya", "slug": slug, "score": 0.7}],
            "copiedTo": ["Emre Kaya/a.jpg"],
            "moved": False,
            "error": None,
            "skipped": None,
            "duplicateOf": None,
            "uncertain": [],
        }
        hooks.on_file(record, 0.2)
        hooks.archive(dst)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "rapor.json").write_text("{}", encoding="utf-8")
        return {
            "scannedAt": "2026-11-06T00:00:00+00:00",
            "complete": True,
            "results": [record],
            "summary": {"files": 2, "matchedFiles": 1, "errors": 0, "people": []},
        }

    monkeypatch.setattr(web, "run_scan", fake_run)
    with TestClient(web.app) as client:
        started = client.post(
            "/api/tara",
            json={"source": str(source), "destination": str(dest), "people": [slug]},
        )
        assert started.status_code == 200, started.text
        for _ in range(100):
            status = client.get("/api/tara").json()
            if status["state"] != "running":
                break
            time.sleep(0.02)
    assert status["state"] == "done"
    assert status["source"] == str(source.resolve())
    assert status["destination"] == str(dest.resolve())
    assert status["people"] == [slug]
    assert status["matched"] == 1 and status["perPerson"] == {slug: 1}
    assert status["lastMatch"]["path"] == "Emre Kaya/a.jpg"
    assert status["lastMatch"]["slug"] == slug
    assert status["lastMatch"]["thumb"].startswith("/api/sonuclar/onizleme?")
    assert status["units"] == {"done": 1.0, "total": 2.0}
    assert status["startedAt"] and status["finishedAt"]
    assert status["elapsedSeconds"] >= 0
    assert status["cancelRequested"] is False


def test_relative_paths_and_pool_exclusion_are_refused(home, tmp_path: Path) -> None:
    _enroll(home, "Emre Kaya")
    with TestClient(web.app) as client:
        relative = client.post("/api/tara", json={"source": "etkinlik", "destination": "cikti"})
        assert relative.status_code == 400
        assert relative.json()["detail"] == "Kaynak klasörün tam yolunu verin."
        rel_pool = client.post("/api/havuz", json={"source": "x", "destination": "y"})
        assert rel_pool.json()["detail"] == "Kaynak klasörün tam yolunu verin."
        source = tmp_path / "etkinlik"
        source.mkdir()
        (source / "a.jpg").write_bytes(b"a")
        with web._pool_lock:
            web._pool["state"] = "running"
        try:
            busy = client.post(
                "/api/tara", json={"source": str(source), "destination": str(tmp_path / "out")}
            )
        finally:
            with web._pool_lock:
                web._pool["state"] = "idle"
        assert busy.status_code == 409
        assert busy.json()["detail"] == "Tekil havuz hazırlanıyor. Bitince taramayı başlatın."
        with web._scan_lock:
            web._scan["state"] = "running"
        try:
            pool = client.post(
                "/api/havuz", json={"source": str(source), "destination": str(tmp_path / "havuz")}
            )
        finally:
            with web._scan_lock:
                web._scan["state"] = "idle"
        assert pool.status_code == 409
        assert pool.json()["detail"] == "Tarama sürüyor. Bitince havuzu hazırlayın."


def test_unwritable_destination_is_refused(home, tmp_path: Path, monkeypatch) -> None:
    _enroll(home, "Emre Kaya")
    source = tmp_path / "etkinlik"
    source.mkdir()
    (source / "a.jpg").write_bytes(b"a")

    def refuse(_dest: Path) -> None:
        raise HTTPException(status_code=400, detail="Çıktı klasörüne yazılamıyor.")

    monkeypatch.setattr(web, "_check_writable", refuse)
    with TestClient(web.app) as client:
        response = client.post(
            "/api/tara", json={"source": str(source), "destination": str(tmp_path / "salt")}
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "Çıktı klasörüne yazılamıyor."


@pytest.mark.parametrize("state", ["cancelled", "error"])
def test_status_keeps_existing_keys(home, state: str) -> None:
    with TestClient(web.app) as client:
        status = client.get("/api/tara").json()
    for key in ("state", "done", "total", "current", "summary", "report", "error", "phase"):
        assert key in status


def test_check_writable_probes_the_destination(tmp_path: Path) -> None:
    dest = tmp_path / "yeni" / "cikti"
    web._check_writable(dest)
    assert dest.is_dir() and not any(dest.iterdir())
    blocked = tmp_path / "dosya"
    blocked.write_bytes(b"x")
    with pytest.raises(HTTPException) as caught:
        web._check_writable(blocked / "alt")
    assert caught.value.detail == "Çıktı klasörüne yazılamıyor."
