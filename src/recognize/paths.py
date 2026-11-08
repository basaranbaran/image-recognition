"""Project directories. Models and enrolled people stay on this machine."""

from __future__ import annotations

import os
from pathlib import Path

from recognize.media import MEDIA_EXTENSIONS

_SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "cikti",
    "modeller",
    "src",
    "tests",
    "veri",
}
POOL_DIR_NAME = "tekil-havuz"


def project_root() -> Path:
    env = os.environ.get("RECOGNIZE_HOME")
    if env:
        return Path(env).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").is_file() and (cwd / "src" / "recognize").is_dir():
        return cwd
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "recognize").is_dir():
            return parent
    return cwd


def data_dir() -> Path:
    path = project_root() / "veri"
    path.mkdir(parents=True, exist_ok=True)
    return path


def people_dir() -> Path:
    path = data_dir() / "kisiler"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = project_root() / "modeller"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_dir() -> Path:
    return project_root() / "cikti"


def default_pool_dir() -> Path:
    return project_root() / POOL_DIR_NAME


def event_source_dir(prefer_pool: bool = True) -> Path | None:
    """Folder that holds the event photos and videos. Code and output dirs are ignored.

    Once the unique pool exists it is the source, since it holds each file once.
    """
    root = project_root()
    if not root.is_dir():
        return None
    pool = root / POOL_DIR_NAME
    if prefer_pool and pool.is_dir() and _media_count(pool):
        return pool.resolve()
    ranked: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or child.name.startswith(".") or child.name in _SKIP_DIRS:
            continue
        if child.name == POOL_DIR_NAME:
            continue
        count = _media_count(child)
        if count:
            ranked.append((count, child))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1].name))
    return ranked[0][1].resolve()


def _media_count(folder: Path) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [
            name for name in dirnames if name not in _SKIP_DIRS and not name.startswith(".")
        ]
        for name in filenames:
            if name.startswith("."):
                continue
            if Path(name).suffix.lower() in MEDIA_EXTENSIONS:
                total += 1
    return total
