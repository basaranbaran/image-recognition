"""Download the Apache-2.0 YuNet and SFace models once, onto this machine."""

from __future__ import annotations

import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

from recognize.errors import AppError

YUNET_NAME = "face_detection_yunet_2023mar.onnx"
SFACE_NAME = "face_recognition_sface_2021dec.onnx"

_YUNET_URLS = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
)
_SFACE_URLS = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx",
)

_MIN_BYTES = {YUNET_NAME: 100_000, SFACE_NAME: 5_000_000}


def ensure_models(directory: Path, log: Callable[[str], None] | None = None) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    yunet = _ensure_one(directory / YUNET_NAME, _YUNET_URLS, log)
    sface = _ensure_one(directory / SFACE_NAME, _SFACE_URLS, log)
    return yunet, sface


def _ensure_one(dest: Path, urls: tuple[str, ...], log: Callable[[str], None] | None) -> Path:
    if _looks_valid(dest):
        return dest
    if log is not None:
        log(f"Model indiriliyor: {dest.name}")
    errors: list[str] = []
    for url in urls:
        try:
            _download(url, dest)
        except (OSError, AppError) as exc:
            errors.append(str(exc))
            continue
        if _looks_valid(dest):
            return dest
        dest.unlink(missing_ok=True)
        errors.append(f"{url} geçersiz dosya döndürdü")
    detail = errors[-1] if errors else "bilinmeyen hata"
    raise AppError(f"Yüz modeli indirilemedi ({dest.name}). {detail}")


def _download(url: str, dest: Path) -> None:
    temporary = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "image-recognition/0.1"})
    try:
        with (
            urllib.request.urlopen(request, timeout=180) as response,
            temporary.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise AppError(f"İndirme başarısız: {exc}") from exc
    temporary.replace(dest)


def _looks_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    size = path.stat().st_size
    if size < _MIN_BYTES[path.name]:
        return False
    head = path.read_bytes()[:80]
    rejected = (b"version https://git-lfs", b"<!DOCTYPE", b"<html")
    if head.startswith(rejected):
        return False
    return True
