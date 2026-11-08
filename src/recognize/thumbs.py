"""Thumbnails of the files a scan copied. Always re-encoded JPEG, never the original bytes."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path

import cv2
from PIL import Image, ImageOps, UnidentifiedImageError

from recognize.errors import AppError
from recognize.media import VIDEO_EXTENSIONS
from recognize.paths import data_dir

SIZES = (320, 640, 1280)
CACHE_LIMIT_BYTES = 512 * 1024 * 1024
CACHE_TARGET_BYTES = 384 * 1024 * 1024
_JPEG_QUALITY = 80

# At most three decodes at once: a 40 MP JPEG is a few hundred MB of pixels while it decodes.
_render_slots = threading.BoundedSemaphore(3)
_inflight_lock = threading.Lock()
_inflight: dict[str, threading.Event] = {}
_warm_stop = threading.Event()
_warm_lock = threading.Lock()
_warm_thread: threading.Thread | None = None


class ThumbError(AppError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status


def cache_dir() -> Path:
    return data_dir() / "onizleme"


def cache_key(path: Path, size: int) -> str:
    """Keyed by inode and mtime, so a rename (reject, undo) keeps the entry valid."""
    stat = path.stat()
    raw = f"{stat.st_dev}:{stat.st_ino}:{stat.st_mtime_ns}:{stat.st_size}:{size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def thumbnail(path: Path, size: int) -> tuple[str, Path]:
    """The cache key and the cached JPEG for ``path`` at ``size``, rendering when needed."""
    if size not in SIZES:
        raise ThumbError(400, "Önizleme boyutu 320, 640 ya da 1280 olabilir.")
    try:
        key = cache_key(path, size)
    except OSError as exc:
        raise ThumbError(
            404, "Dosya artık yerinde değil. Taşınmış ya da silinmiş olabilir."
        ) from exc
    target = cache_dir() / key[:2] / f"{key}.jpg"
    if target.is_file():
        return key, target
    # One render per key: a grid of 26 tiles must not decode the same file three times.
    with _inflight_lock:
        event = _inflight.get(key)
        owner = event is None
        if owner:
            event = threading.Event()
            _inflight[key] = event
    if not owner:
        event.wait(60)
        if target.is_file():
            return key, target
        raise ThumbError(422, "Bu dosyanın önizlemesi oluşturulamadı.")
    try:
        with _render_slots:
            data = render(path, size)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, target)
        _prune_if_needed()
        return key, target
    finally:
        with _inflight_lock:
            _inflight.pop(key, None)
        event.set()


def render(path: Path, size: int) -> bytes:
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return _render_video(path, size)
    return _render_image(path, size)


def _render_image(path: Path, size: int) -> bytes:
    try:
        with Image.open(path) as image:
            if image.format == "JPEG":
                # Draft decode is only for thumbnails; recognition keeps the full decode.
                image.draft("RGB", (size * 2, size * 2))
            upright = ImageOps.exif_transpose(image) or image
            flat = _flatten(upright)
            flat.thumbnail((size, size), Image.Resampling.LANCZOS)
            return _jpeg(flat)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ThumbError(422, "Bu dosyanın önizlemesi oluşturulamadı.") from exc


def _flatten(image: Image.Image) -> Image.Image:
    has_alpha = image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    )
    if has_alpha:
        rgba = image.convert("RGBA")
        white = Image.new("RGB", rgba.size, (255, 255, 255))
        white.paste(rgba, mask=rgba.getchannel("A"))
        return white
    return image.convert("RGB")


def _jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=_JPEG_QUALITY, progressive=True, optimize=True)
    return buffer.getvalue()


def _render_video(path: Path, size: int) -> bytes:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ThumbError(422, "Videodan kare okunamadı. Dosya yarım kopyalanmış olabilir.")
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = count / fps if fps > 0 and count > 0 else 0.0
        frame = None
        for moment in (min(1.0, duration * 0.1), 0.0):
            capture.set(cv2.CAP_PROP_POS_MSEC, moment * 1000)
            ok, candidate = capture.read()
            if ok and candidate is not None:
                frame = candidate
                break
    finally:
        capture.release()
    if frame is None:
        raise ThumbError(422, "Videodan kare okunamadı. Dosya yarım kopyalanmış olabilir.")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    return _jpeg(image)


# ---------------------------------------------------------------- cache housekeeping


def _prune_if_needed() -> None:
    root = cache_dir()
    entries: list[tuple[float, int, Path]] = []
    total = 0
    try:
        for file in root.rglob("*.jpg"):
            stat = file.stat()
            entries.append((stat.st_mtime, stat.st_size, file))
            total += stat.st_size
    except OSError:
        return
    if total <= CACHE_LIMIT_BYTES:
        return
    entries.sort()
    for _mtime, size, file in entries:
        if total <= CACHE_TARGET_BYTES:
            break
        try:
            file.unlink()
            total -= size
        except OSError:
            continue


# ---------------------------------------------------------------- warm-up after a scan


def warm_up(paths: list[Path], size: int = 640, workers: int = 2) -> None:
    """Render the thumbnails of a finished scan in the background, two at a time."""
    stop_warm_up()
    queue = list(paths)
    lock = threading.Lock()

    def work() -> None:
        while not _warm_stop.is_set():
            with lock:
                if not queue:
                    return
                path = queue.pop(0)
            try:
                thumbnail(path, size)
            except (AppError, OSError):
                continue

    def run() -> None:
        threads = [threading.Thread(target=work, name="recognize-thumbs", daemon=True)]
        threads += [
            threading.Thread(target=work, name="recognize-thumbs", daemon=True)
            for _ in range(max(0, workers - 1))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    with _warm_lock:
        _warm_stop.clear()
        global _warm_thread
        _warm_thread = threading.Thread(target=run, name="recognize-warmup", daemon=True)
        _warm_thread.start()


def stop_warm_up() -> None:
    _warm_stop.set()
    with _warm_lock:
        thread = _warm_thread
    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(5)


def warming() -> bool:
    with _warm_lock:
        thread = _warm_thread
    return thread is not None and thread.is_alive()


RenderFn = Callable[[Path, int], bytes]
_log = logging.getLogger("recognize")
