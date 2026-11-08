"""The unique pool keeps one copy of each file and never touches the event folder."""

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from recognize.errors import AppError
from recognize.pool import build_pool


def _jpeg(shade: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (shade, shade, shade)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _event(root: Path) -> Path:
    source = root / "etkinlik"
    for folder in ("FOTOGRAF", "FOTOGRAF 2", "VIDEO"):
        (source / folder).mkdir(parents=True)
    (source / "FOTOGRAF" / "a.jpg").write_bytes(_jpeg(10))
    (source / "FOTOGRAF 2" / "a.jpg").write_bytes(_jpeg(10))
    (source / "FOTOGRAF 2" / "b.jpg").write_bytes(_jpeg(200))
    # Same camera name, different picture: both must survive.
    (source / "FOTOGRAF" / "c.jpg").write_bytes(_jpeg(60))
    (source / "FOTOGRAF 2" / "c.jpg").write_bytes(_jpeg(120))
    (source / "VIDEO" / "bos.mp4").write_bytes(b"")
    (source / "VIDEO" / "xmp.mp4").write_bytes(b"<?xpacket begin=" + b"x" * 64)
    return source


def _tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_pool_keeps_one_of_each_and_leaves_the_source(tmp_path: Path) -> None:
    source = _event(tmp_path)
    before = _tree(source)
    pool = tmp_path / "tekil-havuz"
    report = build_pool(source, pool)

    assert _tree(source) == before
    names = sorted(p.name for p in (pool / "foto").iterdir())
    assert names == ["FOTOGRAF__c.jpg", "a.jpg", "b.jpg", "c.jpg"]
    assert not (pool / "video").exists()
    summary = report["summary"]
    assert summary == {
        "files": 7,
        "unique": 4,
        "photos": 4,
        "videos": 0,
        "duplicates": 1,
        "emptyFiles": 1,
        "broken": 1,
    }
    rows = {row["source"]: row for row in report["files"]}
    assert rows["FOTOGRAF/a.jpg"]["duplicateOf"] == "FOTOGRAF 2/a.jpg"
    assert rows["VIDEO/xmp.mp4"]["status"] == "broken"
    saved = json.loads((pool / "havuz.json").read_text(encoding="utf-8"))
    assert saved["summary"]["unique"] == 4


def test_running_twice_adds_nothing(tmp_path: Path) -> None:
    source = _event(tmp_path)
    pool = tmp_path / "tekil-havuz"
    build_pool(source, pool)
    first = _tree(pool / "foto")
    again = build_pool(source, pool)
    assert _tree(pool / "foto") == first
    assert again["summary"]["unique"] == 4


def test_pool_inside_the_source_is_refused(tmp_path: Path) -> None:
    source = _event(tmp_path)
    with pytest.raises(AppError):
        build_pool(source, source / "havuz")


def test_existing_pool_becomes_the_default_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from recognize.paths import event_source_dir

    monkeypatch.setenv("RECOGNIZE_HOME", str(tmp_path))
    source = _event(tmp_path)
    assert event_source_dir() == source.resolve()
    build_pool(source, tmp_path / "tekil-havuz")
    assert event_source_dir() == (tmp_path / "tekil-havuz").resolve()
    assert event_source_dir(prefer_pool=False) == source.resolve()


def test_windows_pool_copies_without_cp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from recognize import media

    def no_cp(*args, **kwargs):
        raise AssertionError("cp must not run off macOS")

    monkeypatch.setattr(media, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(media.subprocess, "run", no_cp)
    report = build_pool(_event(tmp_path), tmp_path / "tekil-havuz")
    assert report["summary"]["unique"] == 4
    assert (tmp_path / "tekil-havuz" / "foto" / "a.jpg").is_file()


def test_header_only_video_is_broken_in_the_pool(tmp_path: Path, monkeypatch) -> None:
    import cv2

    from recognize import pool

    source = tmp_path / "etkinlik"
    source.mkdir()
    (source / "kapak.mp4").write_bytes(b"\x00" * 4096)

    class _Stub:
        def __init__(self, *_args: object) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def set(self, *_args: object) -> None:
            pass

        def get(self, prop: int) -> float:
            return 850.0 if prop == cv2.CAP_PROP_FRAME_COUNT else 25.0

        def read(self):
            return False, None

        def release(self) -> None:
            pass

    monkeypatch.setattr(pool.cv2, "VideoCapture", _Stub)
    report = build_pool(source, tmp_path / "havuz")
    assert report["summary"]["broken"] == 1
    assert report["files"][0]["status"] == "broken"


def test_pool_survives_a_vanishing_file(tmp_path: Path, monkeypatch) -> None:
    from recognize import pool

    source = tmp_path / "etkinlik"
    source.mkdir()
    (source / "a.jpg").write_bytes(_jpeg(10))
    (source / "kayip.jpg").write_bytes(_jpeg(20))
    real_groups = pool.content_groups

    def vanish(files, should_stop=None):
        (source / "kayip.jpg").unlink()
        return real_groups(files, should_stop)

    monkeypatch.setattr(pool, "content_groups", vanish)
    report = build_pool(source, tmp_path / "havuz")
    rows = {row["source"]: row["status"] for row in report["files"]}
    assert rows == {"a.jpg": "kept", "kayip.jpg": "broken"}
