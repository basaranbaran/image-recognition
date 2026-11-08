"""Small JSON files under ``veri/``: atomic writes and the list of recent scans."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from recognize.paths import data_dir

RECENT_NAME = "son-taramalar.json"
RECENT_LIMIT = 10
ARCHIVE_NAME = "rapor-arsivi"
ARCHIVE_LIMIT = 10


def write_json_atomic(path: Path, payload: object) -> None:
    """Write beside the target and rename it into place; a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> object | None:
    """The parsed file, or None when it is missing or not JSON."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def recent_scans(root: Path | None = None) -> list[dict[str, object]]:
    """Most recent first. Outputs whose report has since vanished are flagged, not dropped."""
    payload = read_json(_recent_path(root))
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    rows: list[dict[str, object]] = []
    for row in items:
        if isinstance(row, dict) and isinstance(row.get("output"), str):
            rows.append({**row, "exists": (Path(row["output"]) / "rapor.json").is_file()})
    return rows


def remember_scan(
    output: Path,
    summary: dict[str, object] | None,
    scanned_at: str,
    *,
    complete: bool = True,
    root: Path | None = None,
) -> None:
    """Put this output at the head of the recent list, so the panel opens on the last scan."""
    key = str(output.expanduser().resolve())
    items = [
        {k: v for k, v in row.items() if k != "exists"}
        for row in recent_scans(root)
        if row.get("output") != key
    ]
    summary = summary or {}
    items.insert(
        0,
        {
            "output": key,
            "scannedAt": scanned_at,
            "complete": complete,
            "files": int(summary.get("files") or 0),
            "matchedFiles": int(summary.get("matchedFiles") or 0),
        },
    )
    write_json_atomic(_recent_path(root), {"items": items[:RECENT_LIMIT]})


def _recent_path(root: Path | None) -> Path:
    base = root if root is not None else data_dir()
    return base / RECENT_NAME


def archive_dir(output: Path, root: Path | None = None) -> Path:
    """Where earlier reports of one output folder are kept: in the app's data, not the user's."""
    base = root if root is not None else data_dir()
    key = hashlib.sha1(str(output.expanduser().resolve()).encode("utf-8")).hexdigest()[:12]
    return base / ARCHIVE_NAME / key


def archive_report(output: Path, root: Path | None = None) -> Path | None:
    """Copy ``<output>/rapor.json`` aside before a new scan overwrites it. Keeps the last 10.

    Returns the archive path, or None when there was nothing to archive.
    """
    source = output / "rapor.json"
    payload = read_json(source)
    if not isinstance(payload, dict):
        return None
    when = str(payload.get("scannedAt") or "")
    stamp = re.sub(r"[^0-9A-Za-z]+", "-", when).strip("-") or datetime.now(UTC).strftime(
        "%Y%m%dT%H%M%S"
    )
    folder = archive_dir(output, root)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{stamp}.json"
    if not target.exists():
        shutil.copy2(source, target)
    kept = sorted(folder.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for stale in kept[ARCHIVE_LIMIT:]:
        stale.unlink(missing_ok=True)
    return target
