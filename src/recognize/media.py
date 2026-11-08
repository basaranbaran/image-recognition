"""Image loading and event-folder listing for the photo and video types in MEDIA_EXTENSIONS."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from recognize.errors import AppError

try:
    # iPhone photos are HEIC; Pillow opens them once this plugin is registered.
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - the dependency is declared, this only guards odd installs
    register_heif_opener = None
else:
    register_heif_opener()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
# OpenCV's bundled FFmpeg reads all of these, including the HEVC .mov an iPhone records.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
ENROLL_EXTENSIONS = IMAGE_EXTENSIONS
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
FORMATS_LINE = "jpg, png, webp, heic fotoğraf; mp4, mov, m4v, avi, mkv video"
# Written by the photo pool at the top of its folder; its photos and videos sit in these buckets.
POOL_REPORT = "havuz.json"
POOL_BUCKETS = {"foto", "video"}


def collect_media(source: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(source):
        dirnames[:] = [name for name in dirnames if not name.startswith(".") and name != "__MACOSX"]
        for filename in filenames:
            if filename.startswith("."):
                continue
            path = Path(dirpath) / filename
            if path.suffix.lower() in MEDIA_EXTENSIONS:
                files.append(path)
    files.sort(key=lambda path: str(path).lower())
    return files


def content_groups(
    files: list[Path],
    should_stop: Callable[[], bool] | None = None,
) -> dict[Path, Path]:
    """Map each byte-identical copy to the first file with the same content.

    Only files that share a size are hashed, so a folder without copies costs one stat per file.
    ``should_stop`` is consulted between size groups; a True answer ends the hashing early.
    """
    by_size: dict[int, list[Path]] = {}
    for path in files:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size:
            by_size.setdefault(size, []).append(path)
    duplicates: dict[Path, Path] = {}
    for group in by_size.values():
        if len(group) < 2:
            continue
        if should_stop is not None and should_stop():
            break
        first_by_digest: dict[str, Path] = {}
        for path in group:
            try:
                digest = _file_digest(path)
            except OSError:
                continue
            first = first_by_digest.setdefault(digest, path)
            if first != path:
                duplicates[path] = first
    return duplicates


def same_content(left: Path, right: Path) -> bool:
    try:
        if left.stat().st_size != right.stat().st_size:
            return False
        return _file_digest(left) == _file_digest(right)
    except OSError:
        return False


def _file_digest(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=20)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def output_name(source: Path, root: Path) -> str:
    try:
        relative = source.relative_to(root)
    except ValueError:
        # A symlinked file resolves outside the folder; only resolve when the plain paths differ.
        relative = source.resolve().relative_to(root.resolve())
    if len(relative.parts) == 1:
        return relative.name
    in_bucket = len(relative.parts) == 2 and relative.parts[0] in POOL_BUCKETS
    if in_bucket and (root / POOL_REPORT).is_file():
        # The pool keeps names unique inside each bucket, so the bucket prefix only adds noise.
        return relative.parts[1]
    return "__".join(relative.parts)


def clone_file(source: Path, target: Path) -> None:
    """Copy with an APFS clone on macOS: the copy shares the blocks, so it costs no extra disk.

    Windows has no ``cp``, and other filesystems cannot clone: those get a plain copy.
    """
    if sys.platform == "darwin":
        try:
            done = subprocess.run(
                ["cp", "-c", "-p", str(source), str(target)],
                capture_output=True,
                check=False,
            )
        except OSError:
            done = None
        if done is not None and done.returncode == 0:
            return
        target.unlink(missing_ok=True)
    shutil.copy2(source, target)


def directories_overlap(source: Path, dest: Path) -> bool:
    left = source.resolve()
    right = dest.resolve()
    if left == right:
        return True
    return _is_relative(right, left) or _is_relative(left, right)


def _is_relative(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def decode_image(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image)
            rgb = np.asarray(image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise AppError("Fotoğraf okunamadı.") from exc
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_image(path: Path) -> np.ndarray:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise AppError(f"Fotoğraf açılamadı: {path.name}") from exc
    if not data:
        raise AppError(f"Dosya boş (0 bayt): {path.name}")
    try:
        return decode_image(data)
    except AppError as exc:
        raise AppError(f"Fotoğraf okunamadı: {path.name}") from exc


def resize_max_edge(image: np.ndarray, max_edge: int) -> np.ndarray:
    height, width = image.shape[:2]
    edge = max(height, width)
    if max_edge <= 0 or edge <= max_edge:
        return image
    scale = max_edge / edge
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def iter_video_frames(path: Path, every_seconds: float) -> Iterator[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise AppError(f"Video açılamadı: {path.name}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(every_seconds, 0.2)
        if fps > 1 and count > 1:
            duration = count / fps
            moment = 0.0
            while moment < duration:
                capture.set(cv2.CAP_PROP_POS_MSEC, moment * 1000)
                ok, frame = capture.read()
                if ok and frame is not None:
                    yield frame
                moment += step
            return
        frame_step = max(int(round((fps or 25) * step)), 1)
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if index % frame_step == 0:
                yield frame
            index += 1
    finally:
        capture.release()
