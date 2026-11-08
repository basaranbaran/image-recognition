"""Download the YuNet (MIT) and SFace (Apache-2.0) models once, onto this machine.

Both come from a pinned OpenCV Zoo commit and are checked by size and SHA-256, so a dropped
connection or a changed upstream file is downloaded again instead of breaking every scan.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from recognize.errors import AppError

YUNET_NAME = "face_detection_yunet_2023mar.onnx"
SFACE_NAME = "face_recognition_sface_2021dec.onnx"

_ZOO = "opencv/opencv_zoo"


@dataclass(frozen=True)
class ModelFile:
    name: str
    folder: str
    commit: str
    size: int
    sha256: str

    @property
    def urls(self) -> tuple[str, ...]:
        path = f"models/{self.folder}/{self.name}"
        return (
            f"https://github.com/{_ZOO}/raw/{self.commit}/{path}",
            f"https://media.githubusercontent.com/media/{_ZOO}/{self.commit}/{path}",
        )


MODELS = {
    YUNET_NAME: ModelFile(
        name=YUNET_NAME,
        folder="face_detection_yunet",
        commit="f12e12798e8314f7c074a6656816c048dcc95b7a",
        size=232_589,
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    SFACE_NAME: ModelFile(
        name=SFACE_NAME,
        folder="face_recognition_sface",
        commit="ba91a3b91d00d76e86540d4013f944bd6b514e39",
        size=38_696_353,
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}


def ensure_models(directory: Path, log: Callable[[str], None] | None = None) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    yunet = _ensure_one(directory, MODELS[YUNET_NAME], log)
    sface = _ensure_one(directory, MODELS[SFACE_NAME], log)
    return yunet, sface


def _ensure_one(directory: Path, model: ModelFile, log: Callable[[str], None] | None) -> Path:
    dest = directory / model.name
    if is_valid(dest, model):
        return dest
    if log is not None:
        megabytes = max(1, round(model.size / 1_000_000))
        log(f"Model indiriliyor: {model.name} ({megabytes} MB, yalnızca ilk seferde)")
    errors: list[str] = []
    for url in model.urls:
        temporary = dest.with_suffix(dest.suffix + ".part")
        try:
            _download(url, temporary)
        except (OSError, AppError) as exc:
            errors.append(str(exc))
            continue
        if is_valid(temporary, model):
            temporary.replace(dest)
            return dest
        temporary.unlink(missing_ok=True)
        errors.append("İndirilen dosya eksik ya da beklenen dosya değil.")
    detail = errors[-1] if errors else "bilinmeyen hata"
    raise AppError(
        f"Yüz modeli indirilemedi ({model.name}). {detail} "
        "İnternet bağlantısını kontrol edip yeniden deneyin."
    )


def _download(url: str, temporary: Path) -> None:
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


def is_valid(path: Path, model: ModelFile) -> bool:
    """Exact size first (cheap), then the hash: a truncated or altered file never loads."""
    try:
        if not path.is_file() or path.stat().st_size != model.size:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return False
    return digest.hexdigest() == model.sha256
