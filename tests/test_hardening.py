"""Fixes from the review after publishing: no overwrites, verified models, safe names."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from starlette.requests import Request

from recognize import models_io, picker, pool, store
from recognize.errors import AppError
from recognize.gallery import Person
from recognize.match import PersonHit, ScanOptions
from recognize.media import decode_image
from recognize.scan import InspectionLike, run_scan
from recognize.textutil import folder_name
from recognize.web import _host_allowed


class _Fake:
    def __init__(self, hits: dict[str, list[PersonHit]]) -> None:
        self._hits = hits

    def inspect_image(self, path: Path, progress=None) -> InspectionLike:
        del progress
        return InspectionLike(self._hits.get(path.name, []), [])

    inspect_video = inspect_image


def _ayse(root: Path) -> Person:
    return Person("Ayşe", "ayse", "", None, np.ones((1, 4), dtype=np.float32), [], root)


HIT = PersonHit("ayse", "Ayşe", 0.7, 1, False, 90.0)


# ---------------------------------------------------------------- no overwrites


def test_a_second_event_never_overwrites_the_first_events_moved_photo(tmp_path: Path) -> None:
    out = tmp_path / "cikti"
    first, second = tmp_path / "etkinlik1", tmp_path / "etkinlik2"
    first.mkdir()
    second.mkdir()
    (first / "IMG_0001.JPG").write_bytes(b"first event")
    (second / "IMG_0001.JPG").write_bytes(b"second event")
    fake = _Fake({"IMG_0001.JPG": [HIT]})

    run_scan(first, out, fake, [_ayse(tmp_path)], ScanOptions(move=True))
    assert not (first / "IMG_0001.JPG").exists()
    report = run_scan(second, out, fake, [_ayse(tmp_path)], ScanOptions(move=False))

    assert (out / "Ayşe" / "IMG_0001.JPG").read_bytes() == b"first event"
    assert (out / "Ayşe" / "IMG_0001 (2).JPG").read_bytes() == b"second event"
    assert report["results"][0]["copiedTo"] == ["Ayşe/IMG_0001 (2).JPG"]


def test_scanning_the_same_event_again_adds_no_numbered_copies(tmp_path: Path) -> None:
    out = tmp_path / "cikti"
    event = tmp_path / "etkinlik"
    event.mkdir()
    (event / "kare.jpg").write_bytes(b"photo")
    fake = _Fake({"kare.jpg": [HIT]})
    run_scan(event, out, fake, [_ayse(tmp_path)], ScanOptions())
    report = run_scan(event, out, fake, [_ayse(tmp_path)], ScanOptions())
    assert sorted(path.name for path in (out / "Ayşe").iterdir()) == ["kare.jpg"]
    assert report["results"][0]["copiedTo"] == ["Ayşe/kare.jpg"]


def test_flattened_names_that_collide_in_one_scan_both_survive_a_move(tmp_path: Path) -> None:
    event = tmp_path / "etkinlik"
    (event / "a").mkdir(parents=True)
    (event / "a" / "b.jpg").write_bytes(b"inside a folder")
    (event / "a__b.jpg").write_bytes(b"at the top")
    out = tmp_path / "cikti"
    fake = _Fake({"b.jpg": [HIT], "a__b.jpg": [HIT]})
    run_scan(event, out, fake, [_ayse(tmp_path)], ScanOptions(move=True))
    kept = sorted(path.read_bytes() for path in (out / "Ayşe").iterdir())
    assert kept == [b"at the top", b"inside a folder"]


# ---------------------------------------------------------------- models


def _model(content: bytes) -> models_io.ModelFile:
    digest = models_io.hashlib.sha256(content).hexdigest()
    return models_io.ModelFile("m.onnx", "x", "c" * 40, len(content), digest)


def test_a_truncated_or_altered_model_is_not_accepted(tmp_path: Path) -> None:
    model = _model(b"abcdef")
    path = tmp_path / "m.onnx"
    path.write_bytes(b"abc")
    assert not models_io.is_valid(path, model)
    path.write_bytes(b"abcdeX")
    assert not models_io.is_valid(path, model)
    path.write_bytes(b"abcdef")
    assert models_io.is_valid(path, model)


def test_a_bad_download_is_retried_from_the_mirror_and_never_left_behind(
    tmp_path: Path, monkeypatch
) -> None:
    good = b"model-bytes"
    model = _model(good)
    tried: list[str] = []

    def fake_download(url: str, temporary: Path) -> None:
        tried.append(url)
        temporary.write_bytes(good[:4] if len(tried) == 1 else good)

    monkeypatch.setattr(models_io, "_download", fake_download)
    path = models_io._ensure_one(tmp_path, model, None)
    assert path.read_bytes() == good
    assert tried == list(model.urls)
    assert not (tmp_path / "m.onnx.part").exists()


def test_model_urls_are_pinned_to_a_commit() -> None:
    for model in models_io.MODELS.values():
        assert all(f"/{model.commit}/" in url for url in model.urls)
        assert all("/main/" not in url for url in model.urls)


# ---------------------------------------------------------------- panel host guard


def _request(host: str, client: str, server: tuple[str, int]) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/kisiler",
        "headers": [(b"host", host.encode())],
        "client": (client, 50000),
        "server": server,
    }
    return Request(scope)


def test_testserver_host_is_refused_on_a_real_socket() -> None:
    # uvicorn with proxy headers would set the client from X-Forwarded-For: testclient.
    assert not _host_allowed(_request("testserver", "testclient", ("127.0.0.1", 8765)))
    assert _host_allowed(_request("testserver", "testclient", ("testserver", 80)))
    assert _host_allowed(_request("127.0.0.1:8765", "127.0.0.1", ("127.0.0.1", 8765)))


# ---------------------------------------------------------------- names, files, Windows


@pytest.mark.parametrize(
    ("name", "folder"),
    [
        ('Ali "Kaptan" Can', "Ali Kaptan Can"),
        ("Ayşe?", "Ayşe"),
        ("Ali: Can", "Ali Can"),
        ("CON", "CON_"),
        ("com1", "com1_"),
        ("Ayşe Yılmaz", "Ayşe Yılmaz"),
    ],
)
def test_person_folders_are_valid_on_windows_and_exfat(name: str, folder: str) -> None:
    assert folder_name(name) == folder


def test_report_write_retries_while_windows_holds_the_file(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []
    real = store.os.replace

    def flaky(source, target):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(32, "being used by another process")
        real(source, target)

    monkeypatch.setattr(store.sys, "platform", "win32")
    monkeypatch.setattr(store.os, "replace", flaky)
    monkeypatch.setattr(store.time, "sleep", lambda _seconds: None)
    store.write_json_atomic(tmp_path / "rapor.json", {"ok": True})
    assert len(calls) == 3
    assert store.read_json(tmp_path / "rapor.json") == {"ok": True}


def test_a_permission_error_outside_windows_is_not_retried(tmp_path: Path, monkeypatch) -> None:
    def denied(source, target):
        raise PermissionError(13, "denied")

    monkeypatch.setattr(store.sys, "platform", "darwin")
    monkeypatch.setattr(store.os, "replace", denied)
    with pytest.raises(PermissionError):
        store.write_json_atomic(tmp_path / "rapor.json", {})


def test_a_giant_image_is_reported_not_crashed(tmp_path: Path, monkeypatch) -> None:
    photo = tmp_path / "panorama.jpg"
    Image.new("RGB", (400, 300), (10, 20, 30)).save(photo)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)  # stands in for a 200 MP panorama
    with pytest.raises(AppError, match="çok büyük"):
        decode_image(photo.read_bytes())
    assert pool._opens(photo) is True


def test_folder_pickers_read_their_answer_as_utf8(monkeypatch) -> None:
    seen: list[dict] = []

    def fake_run(args, **kwargs):
        seen.append(kwargs)
        return subprocess.CompletedProcess(args, 0, stdout="C:\\Etkinlik\\khas hackathon fotoğraf")

    monkeypatch.setattr(picker.subprocess, "run", fake_run)
    assert picker._pick_windows("Seç", "C:\\") == "C:\\Etkinlik\\khas hackathon fotoğraf"
    assert seen[0]["encoding"] == "utf-8"
    assert seen[0]["env"]["PYTHONIOENCODING"] == "utf-8"
