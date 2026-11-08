"""Open a folder, or reveal one file, in Finder, Explorer or the desktop's file manager."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from recognize.errors import AppError

_FAILED = {
    "darwin": "Klasör açılamadı. Yolu kopyalayıp Finder'da açın.",
    "win32": "Klasör açılamadı. Yolu kopyalayıp Gezgin'de açın.",
    "linux": "Klasör açılamadı. Yolu kopyalayıp dosya yöneticisinde açın.",
}


def failure_message() -> str:
    if sys.platform == "darwin":
        return _FAILED["darwin"]
    if sys.platform.startswith("win"):
        return _FAILED["win32"]
    return _FAILED["linux"]


def open_folder(folder: Path) -> None:
    """Show the folder. Never ``shell=True``; the path travels as one argv item."""
    target = str(folder)
    if sys.platform == "darwin":
        _run(["open", target])
        return
    if sys.platform.startswith("win"):
        startfile = getattr(os, "startfile", None)
        if startfile is None:
            raise AppError(failure_message())
        try:
            startfile(target)
        except OSError as exc:
            raise AppError(failure_message()) from exc
        return
    launcher = shutil.which("xdg-open")
    if launcher is None:
        raise AppError(failure_message())
    try:
        subprocess.Popen([launcher, target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as exc:
        raise AppError(failure_message()) from exc


def reveal_file(file: Path) -> None:
    """Open the parent folder with the file selected where the platform can do that."""
    target = str(file)
    if sys.platform == "darwin":
        _run(["open", "-R", target])
        return
    if sys.platform.startswith("win"):
        try:
            # Explorer returns 1 even on success, so its exit code is ignored.
            subprocess.Popen(["explorer", "/select,", target])
        except OSError as exc:
            raise AppError(failure_message()) from exc
        return
    open_folder(file.parent)


def _run(command: list[str]) -> None:
    try:
        done = subprocess.run(command, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AppError(failure_message()) from exc
    if done.returncode != 0:
        raise AppError(failure_message())
