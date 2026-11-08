"""One copy of every distinct photo and video. The event folder is left as it is."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import cv2
from PIL import Image, UnidentifiedImageError

from recognize.errors import AppError
from recognize.media import (
    FORMATS_LINE,
    POOL_REPORT,
    VIDEO_EXTENSIONS,
    clone_file,
    collect_media,
    content_groups,
    directories_overlap,
    same_content,
)
from recognize.store import write_json_atomic

Progress = Callable[[int, int, str], None]

REPORT_NAME = POOL_REPORT


def build_pool(
    source: Path,
    dest: Path,
    on_progress: Progress | None = None,
) -> dict[str, object]:
    """Write each distinct file once into ``dest/foto`` or ``dest/video``.

    Byte-identical copies, empty files and files that do not open are left out
    and listed in the report. Running it again over the same pool adds nothing.
    """
    source = source.expanduser().resolve()
    dest = dest.expanduser().resolve()
    if not source.is_dir():
        raise AppError("Kaynak klasör bulunamadı.")
    if directories_overlap(source, dest):
        raise AppError("Havuz kaynak klasörle iç içe olamaz. Başka bir klasör verin.")
    files = collect_media(source)
    if not files:
        raise AppError(f"Klasörde fotoğraf ya da video yok ({FORMATS_LINE}).")
    total = len(files)
    _progress(on_progress, 0, total, "Aynı dosyalar aranıyor")
    copies = content_groups(files)

    rows: list[dict[str, object]] = []
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(source).as_posix()
        _progress(on_progress, index - 1, total, relative)
        row: dict[str, object] = {"source": relative, "status": "kept", "pool": None}
        original = copies.get(path)
        try:
            size = path.stat().st_size
        except OSError:
            # Vanished or unreadable between the listing and now: not a crash, a broken row.
            size = None
        if size is None:
            row["status"] = "broken"
        elif size == 0:
            row["status"] = "empty"
        elif original is not None:
            row["status"] = "duplicate"
            row["duplicateOf"] = original.relative_to(source).as_posix()
        elif not _opens(path):
            row["status"] = "broken"
        else:
            folder = dest / ("video" if path.suffix.lower() in VIDEO_EXTENSIONS else "foto")
            target = _place(path, folder, relative)
            row["pool"] = target.relative_to(dest).as_posix()
        rows.append(row)

    _progress(on_progress, total, total, "")
    counts = {status: 0 for status in ("kept", "duplicate", "empty", "broken")}
    for row in rows:
        counts[str(row["status"])] += 1
    kept = [row for row in rows if row["status"] == "kept"]
    report: dict[str, object] = {
        "createdAt": datetime.now(UTC).isoformat(),
        "source": str(source),
        "destination": str(dest),
        "summary": {
            "files": total,
            "unique": counts["kept"],
            "photos": sum(1 for row in kept if str(row["pool"]).startswith("foto/")),
            "videos": sum(1 for row in kept if str(row["pool"]).startswith("video/")),
            "duplicates": counts["duplicate"],
            "emptyFiles": counts["empty"],
            "broken": counts["broken"],
        },
        "files": rows,
    }
    write_json_atomic(dest / REPORT_NAME, report)
    return report


def _opens(path: Path) -> bool:
    """A 61 KB XMP packet named .mp4 is not a video, and neither is a header without frames."""
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return video_decodes(path)
    try:
        with Image.open(path) as image:
            image.size  # noqa: B018 - reading the header is the check
    except Image.DecompressionBombError:
        return True
    except (UnidentifiedImageError, OSError):
        return False
    return True


def video_decodes(path: Path) -> bool:
    """True when the container opens and at least one frame comes out of it."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened() or capture.get(cv2.CAP_PROP_FRAME_COUNT) <= 0:
            return False
        for moment_ms in (0.0, 1000.0):
            capture.set(cv2.CAP_PROP_POS_MSEC, moment_ms)
            ok, frame = capture.read()
            if ok and frame is not None:
                return True
        return False
    finally:
        capture.release()


def _place(path: Path, folder: Path, relative: str) -> Path:
    """Keep the camera's file name; prefix the source folders only when two files collide."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in (path.name, relative.replace("/", "__")):
        target = folder / name
        if not target.exists():
            clone_file(path, target)
            return target
        if same_content(path, target):
            return target
    raise AppError(f"Havuzda {path.name} için yer açılamadı.")


def _progress(callback: Progress | None, done: int, total: int, label: str) -> None:
    if callback is not None:
        callback(done, total, label)
