"""Native folder picker for the local panel: Finder on macOS, Explorer on Windows.

A browser never hands a page the absolute path of a folder, but the panel's
server runs on the same machine, so it opens the system's own dialog.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from recognize.errors import AppError

# argv keeps the prompt and the start folder out of the script text, so no quoting is needed.
_MAC_SCRIPT = (
    "on run argv",
    "activate",
    "set startFolder to POSIX file (item 2 of argv)",
    "return POSIX path of (choose folder with prompt (item 1 of argv)"
    " default location startFolder)",
    "end run",
)
_MAC_CANCELLED = "-128"

# Tk's chooser is the Vista-style Explorer dialog on Windows.
_TK_SCRIPT = """
import sys
import tkinter
from tkinter import filedialog

root = tkinter.Tk()
root.withdraw()
root.attributes("-topmost", True)
root.update()
chosen = filedialog.askdirectory(
    parent=root, title=sys.argv[1], initialdir=sys.argv[2], mustexist=False
)
sys.stdout.write(chosen or "")
root.destroy()
"""

_POWERSHELL = (
    "Add-Type -AssemblyName System.Windows.Forms;"
    "$d = New-Object System.Windows.Forms.FolderBrowserDialog;"
    "$d.Description = $env:RECOGNIZE_PICK_TITLE;"
    "$d.SelectedPath = $env:RECOGNIZE_PICK_START;"
    "$d.ShowNewFolderButton = $true;"
    "if ($d.ShowDialog() -eq 'OK') { [Console]::Out.Write($d.SelectedPath) }"
)
# Without this the chosen path comes back in the ANSI/OEM code page: "fotoğraf" turns "foto?raf".
_POWERSHELL_UTF8 = "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false);"


def platform_name() -> str:
    if sys.platform == "darwin":
        return "mac"
    if sys.platform.startswith("win"):
        return "windows"
    return "linux"


def pick_folder(title: str, current: str | None = None) -> str | None:
    """Open the system folder dialog. Returns the chosen folder, or None when cancelled."""
    start = str(_existing_folder(current))
    platform = platform_name()
    if platform == "mac":
        chosen = _pick_mac(title, start)
    elif platform == "windows":
        chosen = _pick_windows(title, start)
    else:
        chosen = _pick_linux(title, start)
    if not chosen:
        return None
    return str(Path(chosen).expanduser().resolve())


def _existing_folder(current: str | None) -> Path:
    """The dialog must open on a folder that exists; walk up until one does."""
    if current:
        path = Path(current).expanduser()
        for candidate in (path, *path.parents):
            if candidate.is_dir():
                return candidate
    return Path.home()


def _pick_mac(title: str, start: str) -> str | None:
    command = ["osascript"]
    for line in _MAC_SCRIPT:
        command += ["-e", line]
    done = subprocess.run([*command, title, start], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        if _MAC_CANCELLED in done.stderr:
            return None
        raise AppError("Finder penceresi açılamadı. Yolu elle yazın.")
    chosen = done.stdout.strip()
    return chosen.rstrip("/") or "/"


def _pick_windows(title: str, start: str) -> str | None:
    done = _run_tk(title, start)
    if done is not None and done.returncode == 0:
        return done.stdout.strip() or None
    shell = shutil.which("powershell.exe") or shutil.which("powershell")
    if shell is None:
        raise AppError("Klasör penceresi açılamadı. Yolu elle yazın.")
    env = {**os.environ, "RECOGNIZE_PICK_TITLE": title, "RECOGNIZE_PICK_START": start}
    done = subprocess.run(
        [shell, "-NoProfile", "-STA", "-Command", _POWERSHELL_UTF8 + _POWERSHELL],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    if done.returncode != 0:
        raise AppError("Klasör penceresi açılamadı. Yolu elle yazın.")
    return done.stdout.strip() or None


def _pick_linux(title: str, start: str) -> str | None:
    zenity = shutil.which("zenity")
    if zenity is not None:
        done = subprocess.run(
            [zenity, "--file-selection", "--directory", f"--title={title}", f"--filename={start}/"],
            capture_output=True,
            text=True,
            check=False,
        )
        # zenity exits 1 when the user cancels; the output is empty then.
        return done.stdout.strip() or None
    done = _run_tk(title, start)
    if done is None or done.returncode != 0:
        raise AppError("Klasör penceresi açılamadı. Yolu elle yazın.")
    return done.stdout.strip() or None


def _run_tk(title: str, start: str) -> subprocess.CompletedProcess[str] | None:
    """Tk runs in its own process: it must own the main thread, which uvicorn already has."""
    try:
        return subprocess.run(
            [sys.executable, "-c", _TK_SCRIPT, title, start],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            check=False,
        )
    except OSError:
        return None
