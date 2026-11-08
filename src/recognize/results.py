"""Reading a scan report back for the panel, and correcting it ("Bu kişi değil").

Everything here works on ``<output>/rapor.json`` and the files it lists. Nothing is ever
read from ``report.source``; the only file operation is a rename inside the output folder.
"""

from __future__ import annotations

import logging
import os
import threading
import unicodedata
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode

from recognize.errors import AppError
from recognize.media import MEDIA_EXTENSIONS, VIDEO_EXTENSIONS
from recognize.paths import default_output_dir
from recognize.scan import summarize_results
from recognize.store import read_json, recent_scans, write_json_atomic
from recognize.textutil import folder_name

REJECT_DIR = "_Yanlış eşleşmeler"
ASIDE_DIR = "_Önceki taramalar"
REVIEW_MARGIN = 0.02
THUMB_SIZES = (320, 640, 1280)
_CACHE_LIMIT = 8

_cache_lock = threading.Lock()
_cache: dict[Path, tuple[tuple[int, int], dict[str, object]]] = {}
# One correction at a time: the report is re-read, changed and rewritten under this lock.
_report_lock = threading.Lock()


class ResultsError(AppError):
    """A refusal with the HTTP status the panel should answer with."""

    def __init__(self, status: int, detail: str, **extra: object) -> None:
        super().__init__(detail)
        self.status = status
        self.extra = extra


# ---------------------------------------------------------------- output folder and report


def resolve_output(raw: str | None) -> Path:
    """The folder a request means. Empty means the most recent scan, else the default."""
    if raw is None or not raw.strip():
        return default_output()
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        raise ResultsError(400, "Çıktı klasörünün tam yolunu verin.")
    if not path.is_dir():
        raise ResultsError(404, "Çıktı klasörü bulunamadı.")
    return path.resolve()


def default_output() -> Path:
    for row in recent_scans():
        if row.get("exists"):
            return Path(str(row["output"]))
    return default_output_dir().resolve()


def report_path(output: Path) -> Path:
    return output / "rapor.json"


def load_report(output: Path) -> dict[str, object]:
    """The parsed report, cached by path, mtime and size. Raises ``ResultsError``."""
    path = report_path(output)
    try:
        stat = path.stat()
    except OSError as exc:
        raise ResultsError(404, "Bu klasörde tarama raporu yok. Önce ayırmayı başlatın.") from exc
    stamp = (stat.st_mtime_ns, stat.st_size)
    with _cache_lock:
        cached = _cache.get(path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
    report = read_report(output)
    with _cache_lock:
        if len(_cache) >= _CACHE_LIMIT and path not in _cache:
            _cache.pop(next(iter(_cache)))
        _cache[path] = (stamp, report)
    return report


def read_report(output: Path) -> dict[str, object]:
    """Always from disk; corrections use this so they never work on a stale copy."""
    path = report_path(output)
    if not path.is_file():
        raise ResultsError(404, "Bu klasörde tarama raporu yok. Önce ayırmayı başlatın.")
    payload = read_json(path)
    if payload is None:
        raise ResultsError(422, "Tarama raporu okunamadı. rapor.json bozuk olabilir.")
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ResultsError(422, "Bu rapor.json bir tarama raporu değil.")
    return payload


# ---------------------------------------------------------------- listed files


def normalise_path(raw: str) -> str:
    """NFC with ``/`` separators. Anything that could leave the output folder is refused."""
    text = unicodedata.normalize("NFC", str(raw or "")).replace("\\", "/")
    if not text or text.startswith("/") or "\x00" in text:
        raise ResultsError(404, "Bu dosya tarama raporunda yok.")
    parts = PurePosixPath(text).parts
    if any(part in {"..", "."} for part in parts) or not parts:
        raise ResultsError(404, "Bu dosya tarama raporunda yok.")
    return text


def allowed_paths(report: dict[str, object]) -> set[str]:
    """Every file the report put under its destination: copies plus moved-aside rejects."""
    allowed: set[str] = set()
    for record in _records(report):
        for item in record.get("copiedTo") or []:
            if isinstance(item, str):
                allowed.add(unicodedata.normalize("NFC", item.replace("\\", "/")))
        for row in record.get("rejected") or []:
            if isinstance(row, dict) and isinstance(row.get("movedTo"), str):
                allowed.add(unicodedata.normalize("NFC", row["movedTo"].replace("\\", "/")))
    return allowed


def listed_file(
    output: Path,
    report: dict[str, object],
    raw: str,
    extra: set[str] | None = None,
) -> Path:
    """The on-disk file behind a report path, or a 404 ``ResultsError``.

    ``extra`` holds paths a running scan has already written but not yet checkpointed.
    """
    path = normalise_path(raw)
    allowed = allowed_paths(report)
    if extra:
        allowed |= {unicodedata.normalize("NFC", item.replace("\\", "/")) for item in extra}
    if path not in allowed:
        raise ResultsError(404, "Bu dosya tarama raporunda yok.")
    if PurePosixPath(path).suffix.lower() not in MEDIA_EXTENSIONS:
        raise ResultsError(404, "Bu dosya tarama raporunda yok.")
    root = output.resolve()
    try:
        target = (root / path).resolve(strict=True)
    except OSError as exc:
        raise ResultsError(
            404, "Dosya artık yerinde değil. Taşınmış ya da silinmiş olabilir."
        ) from exc
    if root not in target.parents or not target.is_file():
        raise ResultsError(404, "Bu dosya tarama raporunda yok.")
    return target


def file_version(path: Path) -> str | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return f"{stat.st_mtime_ns:x}-{stat.st_size:x}"


def thumb_url(output: Path, path: str, version: str | None = None, size: int | None = None) -> str:
    query: dict[str, str] = {"output": str(output), "path": path}
    if size is not None:
        query["size"] = str(size)
    if version:
        query["v"] = version
    return "/api/sonuclar/onizleme?" + urlencode(query)


# ---------------------------------------------------------------- the listing


def build_results(
    output: Path,
    report: dict[str, object],
    *,
    person: str | None = None,
    scan_running: bool = False,
    enrolled: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Group the report's copies by person, with on-disk facts the UI needs.

    ``enrolled`` maps slug to ``{"cover": url}`` for people who are still enrolled.
    """
    output = output.resolve()
    threshold = _float(report.get("threshold"), 0.5)
    people_rows = _people_rows(report)
    by_folder = {row["folder"]: row for row in people_rows}
    files: dict[str, list[dict[str, object]]] = {row["slug"]: [] for row in people_rows}
    rejected: dict[str, list[dict[str, object]]] = {row["slug"]: [] for row in people_rows}
    unmatched: dict[str, list[dict[str, object]]] = {"errors": [], "skipped": [], "uncertain": []}
    copies_of: dict[str, int] = {}
    not_copied = 0
    missing_total = 0

    # Leftovers are read from disk, so a move-aside or a manual change shows at once. A running
    # or unfinished scan has not copied everything yet, so nothing is called a leftover then.
    settled = report.get("complete") is not False and not scan_running
    lingering = leftover_map(output, report) if settled else {}

    records = _records(report)
    for record in records:
        twin = record.get("duplicateOf")
        if isinstance(twin, str) and twin:
            copies_of[twin] = copies_of.get(twin, 0) + 1
    for record in records:
        source = str(record.get("source") or "")
        kind = "video" if record.get("kind") == "video" else "image"
        if record.get("duplicateOf"):
            continue
        error = record.get("error")
        if isinstance(error, str) and error:
            unmatched["errors"].append({"source": source, "kind": kind, "error": error})
        skipped = record.get("skipped")
        if isinstance(skipped, str) and skipped:
            unmatched["skipped"].append({"source": source, "kind": kind, "note": skipped})
        matches = [row for row in record.get("matches") or [] if isinstance(row, dict)]
        uncertain = [row for row in record.get("uncertain") or [] if isinstance(row, dict)]
        if not matches and uncertain:
            unmatched["uncertain"].append(
                {
                    "source": source,
                    "kind": kind,
                    "candidates": [
                        {
                            "slug": str(row.get("slug") or ""),
                            "name": str(row.get("name") or ""),
                            "score": _float(row.get("score"), 0.0),
                            "facePx": _optional_float(row.get("facePx")),
                        }
                        for row in uncertain
                    ],
                }
            )
        match_slugs = [str(row.get("slug") or "") for row in matches]
        copied = [item for item in record.get("copiedTo") or [] if isinstance(item, str)]
        seen_slugs: set[str] = set()
        for item in copied:
            path = unicodedata.normalize("NFC", item.replace("\\", "/"))
            folder = PurePosixPath(path).parts[0] if PurePosixPath(path).parts else ""
            row = by_folder.get(folder)
            if row is None:
                continue
            slug = str(row["slug"])
            seen_slugs.add(slug)
            match = next((m for m in matches if str(m.get("slug")) == slug), None)
            target = output / path
            exists = target.is_file()
            version = file_version(target) if exists else None
            if not exists:
                missing_total += 1
            score = _optional_float(match.get("score")) if match else None
            review: list[str] = []
            if match:
                if match.get("smallFace"):
                    review.append("smallFace")
                if match.get("learned"):
                    review.append("learned")
                if match.get("clothingAssisted"):
                    review.append("clothingAssisted")
                if score is not None and score < threshold + REVIEW_MARGIN:
                    review.append("nearThreshold")
            files[slug].append(
                {
                    "path": path,
                    "name": PurePosixPath(path).name,
                    "source": source,
                    "kind": kind,
                    "score": score,
                    "facePx": _optional_float(match.get("facePx")) if match else None,
                    "smallFace": bool(match.get("smallFace")) if match else False,
                    "learned": bool(match.get("learned")) if match else False,
                    "clothingAssisted": bool(match.get("clothingAssisted")) if match else False,
                    "samples": _optional_int(match.get("samples")) if match else None,
                    "box": _box(match.get("box")) if match else None,
                    "review": review,
                    "with": [other for other in match_slugs if other and other != slug],
                    "copies": copies_of.get(source, 0),
                    "exists": exists,
                    "bytes": _size(target) if exists else None,
                    "thumb": thumb_url(output, path, version) if exists else None,
                }
            )
        for slug in match_slugs:
            if slug and slug in files and slug not in seen_slugs:
                not_copied += 1
        for row in record.get("rejected") or []:
            if not isinstance(row, dict):
                continue
            slug = str(row.get("slug") or "")
            moved = row.get("movedTo")
            if slug not in rejected or not isinstance(moved, str):
                continue
            moved = unicodedata.normalize("NFC", moved.replace("\\", "/"))
            target = output / moved
            exists = target.is_file()
            rejected[slug].append(
                {
                    "path": moved,
                    "originalPath": str(row.get("path") or ""),
                    "source": source,
                    "kind": kind,
                    "score": _optional_float(row.get("score")),
                    "facePx": _optional_float(row.get("facePx")),
                    "rejectedAt": row.get("rejectedAt"),
                    "exists": exists,
                    "thumb": thumb_url(output, moved, file_version(target)) if exists else None,
                }
            )

    enrolled = enrolled or {}
    summary_people = {
        str(row.get("slug")): row
        for row in (report.get("summary") or {}).get("people") or []
        if isinstance(row, dict)
    }
    people_out: list[dict[str, object]] = []
    for row in people_rows:
        slug = str(row["slug"])
        if person is not None and slug != person:
            continue
        own = sorted(files[slug], key=lambda item: str(item["path"]))
        own_rejected = sorted(rejected[slug], key=lambda item: str(item["path"]))
        folder_dir = output / str(row["folder"])
        known = enrolled.get(slug)
        people_out.append(
            {
                "name": row["name"],
                "slug": slug,
                "folder": row["folder"],
                "folderPath": str(folder_dir),
                "folderExists": folder_dir.is_dir(),
                "enrolled": known is not None,
                "cover": None if known is None else known.get("cover"),
                "clothingLabel": (summary_people.get(slug) or {}).get("clothingLabel"),
                "leftovers": [_nfc(name) for name in lingering.get(str(row["folder"]), [])],
                "counts": {
                    "files": len(own),
                    "images": sum(1 for item in own if item["kind"] == "image"),
                    "videos": sum(1 for item in own if item["kind"] == "video"),
                    "smallFaces": sum(1 for item in own if item["smallFace"]),
                    "learned": sum(1 for item in own if item["learned"]),
                    "clothingAssisted": sum(1 for item in own if item["clothingAssisted"]),
                    "review": sum(1 for item in own if item["review"]),
                    "rejected": len(own_rejected),
                    "missing": sum(1 for item in own if not item["exists"]),
                    "leftovers": len(lingering.get(str(row["folder"]), [])),
                },
                "files": own,
                "rejected": own_rejected,
            }
        )
    if person is not None and not people_out:
        raise ResultsError(404, "Bu kişi bu taramada yok.")

    summary = dict(report.get("summary") or {})
    summary.setdefault("rejected", sum(len(rows) for rows in rejected.values()))
    summary["missing"] = missing_total
    summary["notCopied"] = not_copied
    summary["leftovers"] = sum(len(names) for names in lingering.values())
    source_dir = str(report.get("source") or "")
    destination = str(report.get("destination") or "")
    relocated = False
    if destination:
        try:
            relocated = Path(destination).expanduser().resolve() != output
        except OSError:
            relocated = True
    progress = report.get("progress")
    return {
        "output": str(output),
        "reportPath": str(report_path(output)),
        "archive": None,
        "scannedAt": report.get("scannedAt"),
        "stoppedAt": report.get("stoppedAt"),
        "complete": report.get("complete") is not False,
        "progress": progress if isinstance(progress, dict) else None,
        "relocated": relocated,
        "scanRunning": scan_running,
        "source": source_dir,
        "sourceExists": bool(source_dir) and Path(source_dir).is_dir(),
        "move": bool(report.get("move")),
        "threshold": threshold,
        "sampleSeconds": _float(report.get("sampleSeconds"), 0.5),
        "summary": summary,
        "people": people_out,
        "unmatched": unmatched,
        "corrections": len([c for c in report.get("corrections") or [] if isinstance(c, dict)]),
    }


# ---------------------------------------------------------------- removed people


def hide_removed_people(report: dict[str, object], present: set[str]) -> dict[str, object]:
    """The report as the panel shows it: people whose gallery folder is gone are left out.

    A person deleted in the panel keeps their photos in the output folder, but nothing of
    theirs is shown. Their matches, copies and rejects drop out of this copy, and the summary
    is counted again. The stored report is not changed.
    """
    rows = _people_rows(report)
    gone = {str(row["slug"]) for row in rows if row["slug"] not in present}
    if not gone:
        return report
    gone_folders = {_nfc(str(row["folder"])) for row in rows if row["slug"] in gone}
    shown_rows = [row for row in rows if row["slug"] not in gone]
    shown = dict(report)
    records: list[dict[str, object]] = []
    for record in _records(report):
        cleaned = dict(record)
        for key in ("matches", "uncertain", "rejected"):
            cleaned[key] = [row for row in record.get(key) or [] if not _names_gone(row, gone)]
        cleaned["copiedTo"] = [
            item for item in record.get("copiedTo") or [] if not _in_folders(item, gone_folders)
        ]
        records.append(cleaned)
    shown["results"] = records
    shown["people"] = [
        row
        for row in report.get("people") or []
        if not (isinstance(row, dict) and str(row.get("slug")) in gone)
    ]
    shown["summary"] = summarize_results(records, _summary_rows(shown, shown_rows))
    return shown


def _names_gone(row: object, gone: set[str]) -> bool:
    return isinstance(row, dict) and str(row.get("slug") or "") in gone


def _in_folders(item: object, folders: set[str]) -> bool:
    if not isinstance(item, str):
        return False
    parts = PurePosixPath(_nfc(item)).parts
    return bool(parts) and parts[0] in folders


# ---------------------------------------------------------------- leftovers


def leftover_map(
    output: Path,
    report: dict[str, object],
    slugs: set[str] | None = None,
) -> dict[str, list[str]]:
    """Per person folder, the media files that this report did not copy there.

    These come from earlier runs into the same output. Names are the raw on-disk names,
    so a move finds the file again; compare them through ``_nfc``.
    """
    copied: dict[str, set[str]] = {}
    for record in _records(report):
        for item in record.get("copiedTo") or []:
            path = PurePosixPath(_nfc(item)) if isinstance(item, str) else None
            if path is not None and len(path.parts) == 2:
                copied.setdefault(path.parts[0], set()).add(path.name)
    found: dict[str, list[str]] = {}
    for row in _people_rows(report):
        if slugs is not None and row["slug"] not in slugs:
            continue
        folder = str(row["folder"])
        folder_dir = output / folder
        if not folder_dir.is_dir():
            continue
        known = copied.get(folder, set())
        names: list[str] = []
        with os.scandir(folder_dir) as entries:
            for entry in entries:
                if entry.name.startswith(".") or _nfc(entry.name) in known:
                    continue
                if PurePosixPath(entry.name).suffix.lower() not in MEDIA_EXTENSIONS:
                    continue
                if entry.is_file():
                    names.append(entry.name)
        if names:
            found[folder] = sorted(names, key=str.lower)
    return found


def move_leftovers(
    output: Path,
    slug: str | None = None,
    *,
    scan_running: bool = False,
    present: set[str] | None = None,
) -> dict[str, object]:
    """Move earlier-run files into ``_Önceki taramalar``, keeping each folder. Never deletes."""
    if scan_running:
        raise ResultsError(409, "Bu klasöre tarama sürüyor. Bitince yeniden deneyin.")
    output = output.resolve()
    with _report_lock:
        report = read_report(output)
        if present is not None:
            report = hide_removed_people(report, present)
        if report.get("complete") is False:
            raise ResultsError(409, "Tarama tamamlanmadı. Önce taramayı bitirin.")
        rows = _people_rows(report)
        if slug is not None:
            rows = [row for row in rows if row["slug"] == slug]
            if not rows:
                raise ResultsError(404, "Bu kişi bu taramada yok.")
        found = leftover_map(output, report, {str(row["slug"]) for row in rows})
        moved = 0
        for folder, names in found.items():
            for name in names:
                target = _free_name(output / ASIDE_DIR / folder, _nfc(name))
                _move(output / folder / name, target)
                moved += 1
    return {"moved": moved}


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", str(text).replace("\\", "/"))


# ---------------------------------------------------------------- corrections


def reject_match(
    output: Path,
    raw_path: str,
    slug: str | None = None,
    *,
    scan_running: bool = False,
) -> dict[str, object]:
    """Move one copy out of a person's folder into ``_Yanlış eşleşmeler`` and fix the report."""
    if scan_running:
        raise ResultsError(409, "Bu klasöre tarama sürüyor. Bitince yeniden deneyin.")
    output = output.resolve()
    path = normalise_path(raw_path)
    with _report_lock:
        report = read_report(output)
        record = _record_with_copy(report, path)
        if record is None:
            earlier = _rejected_row(report, original=path)
            if earlier is not None and (output / str(earlier[1]["movedTo"])).is_file():
                return _correction_result(report, path, str(earlier[1]["movedTo"]), True)
            raise ResultsError(404, "Bu dosya tarama raporunda yok.")
        folder = PurePosixPath(path).parts[0]
        people_rows = _people_rows(report)
        owner = next((row for row in people_rows if row["folder"] == folder), None)
        if owner is None:
            raise ResultsError(404, "Bu dosya tarama raporunda yok.")
        if slug is not None and slug != owner["slug"]:
            raise ResultsError(400, "Dosya bu kişinin klasöründe değil.")
        source_file = output / path
        if not source_file.is_file():
            raise ResultsError(
                404, "Dosya kişi klasöründe yok; elle taşınmış ya da silinmiş olabilir."
            )
        matches = [row for row in record.get("matches") or [] if isinstance(row, dict)]
        match = next((row for row in matches if str(row.get("slug")) == owner["slug"]), None)
        if match is None:
            match = {"name": owner["name"], "slug": owner["slug"]}
        target_dir = output / REJECT_DIR / folder
        target = _free_name(target_dir, PurePosixPath(path).name)
        _move(source_file, target)
        moved_to = target.relative_to(output).as_posix()
        now = datetime.now(UTC).isoformat()
        record["matches"] = [row for row in matches if row is not match]
        record["copiedTo"] = [
            item for item in record.get("copiedTo") or [] if not _same_path(item, path)
        ]
        rows = [row for row in record.get("rejected") or [] if isinstance(row, dict)]
        rows.append({**match, "path": path, "movedTo": moved_to, "rejectedAt": now})
        record["rejected"] = rows
        _sync_duplicates(report, record)
        _recount(report, people_rows)
        _append_correction(report, "reject", path, str(owner["slug"]), moved_to, now)
        try:
            write_json_atomic(report_path(output), report)
        except OSError as exc:
            logging.exception("Could not rewrite the report after a reject")
            _move(target, source_file)
            raise ResultsError(500, "Rapor güncellenemedi; dosya yerine geri kondu.") from exc
        return _correction_result(report, path, moved_to, False)


def undo_reject(output: Path, raw_path: str, *, scan_running: bool = False) -> dict[str, object]:
    """Put a moved-aside copy back, keyed by its ``movedTo`` path."""
    if scan_running:
        raise ResultsError(409, "Bu klasöre tarama sürüyor. Bitince yeniden deneyin.")
    output = output.resolve()
    moved_to = normalise_path(raw_path)
    with _report_lock:
        report = read_report(output)
        found = _rejected_row(report, moved=moved_to)
        if found is None:
            raise ResultsError(404, "Geri alınacak bir ayıklama yok.")
        record, row = found[0], found[1]
        original = normalise_path(str(row.get("path") or ""))
        source_file = output / moved_to
        if not source_file.is_file():
            raise ResultsError(404, "Ayıklanan dosya yerinde değil.")
        target = output / original
        if target.exists():
            raise ResultsError(409, "Kişi klasöründe aynı adlı bir dosya var; geri alınamadı.")
        _move(source_file, target)
        match = {k: v for k, v in row.items() if k not in {"path", "movedTo", "rejectedAt"}}
        matches = [m for m in record.get("matches") or [] if isinstance(m, dict)]
        matches.append(match)
        record["matches"] = matches
        copied = [item for item in record.get("copiedTo") or [] if isinstance(item, str)]
        copied.append(original)
        record["copiedTo"] = copied
        record["rejected"] = [
            r for r in record.get("rejected") or [] if isinstance(r, dict) and r is not row
        ]
        _sync_duplicates(report, record)
        people_rows = _people_rows(report)
        _recount(report, people_rows)
        now = datetime.now(UTC).isoformat()
        _append_correction(report, "undo", original, str(match.get("slug") or ""), moved_to, now)
        try:
            write_json_atomic(report_path(output), report)
        except OSError as exc:
            logging.exception("Could not rewrite the report after an undo")
            _move(target, source_file)
            raise ResultsError(500, "Rapor güncellenemedi; dosya yerine geri kondu.") from exc
        person = _person_counts(report, str(match.get("slug") or ""))
        return {
            "path": original,
            "restoredFrom": moved_to,
            "person": person,
            "summary": report.get("summary"),
        }


# ---------------------------------------------------------------- helpers


def _records(report: dict[str, object]) -> list[dict[str, object]]:
    results = report.get("results")
    if not isinstance(results, list):
        return []
    return [row for row in results if isinstance(row, dict)]


def _people_rows(report: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in report.get("people") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        slug = str(item.get("slug") or "")
        if not name or not slug:
            continue
        try:
            folder = folder_name(name)
        except AppError:
            folder = name
        rows.append({"name": name, "slug": slug, "folder": folder})
    return rows


def _summary_rows(report: dict[str, object], people_rows: list[dict[str, object]]):
    labels = {
        str(row.get("slug")): row.get("clothingLabel")
        for row in (report.get("summary") or {}).get("people") or []
        if isinstance(row, dict)
    }
    return [
        {"name": row["name"], "slug": row["slug"], "clothingLabel": labels.get(str(row["slug"]))}
        for row in people_rows
    ]


def _recount(report: dict[str, object], people_rows: list[dict[str, object]]) -> None:
    report["summary"] = summarize_results(_records(report), _summary_rows(report, people_rows))


def _record_with_copy(report: dict[str, object], path: str) -> dict[str, object] | None:
    for record in _records(report):
        for item in record.get("copiedTo") or []:
            if _same_path(item, path):
                return record
    return None


def _same_path(item: object, path: str) -> bool:
    if not isinstance(item, str):
        return False
    return unicodedata.normalize("NFC", item.replace("\\", "/")) == path


def _rejected_row(
    report: dict[str, object], *, moved: str | None = None, original: str | None = None
) -> tuple[dict[str, object], dict[str, object]] | None:
    for record in _records(report):
        for row in record.get("rejected") or []:
            if not isinstance(row, dict):
                continue
            moved_to = unicodedata.normalize("NFC", str(row.get("movedTo") or "")).replace(
                "\\", "/"
            )
            path = unicodedata.normalize("NFC", str(row.get("path") or "")).replace("\\", "/")
            if moved is not None and moved_to == moved:
                return record, row
            if original is not None and path == original:
                return record, row
    return None


def _sync_duplicates(report: dict[str, object], record: dict[str, object]) -> None:
    source = record.get("source")
    for twin in _records(report):
        if twin.get("duplicateOf") == source:
            twin["matches"] = list(record.get("matches") or [])
            twin["uncertain"] = list(record.get("uncertain") or [])


def _append_correction(
    report: dict[str, object], action: str, path: str, slug: str, moved_to: str, at: str
) -> None:
    rows = [row for row in report.get("corrections") or [] if isinstance(row, dict)]
    rows.append({"at": at, "action": action, "path": path, "slug": slug, "movedTo": moved_to})
    report["corrections"] = rows


def _correction_result(
    report: dict[str, object], path: str, moved_to: str, already: bool
) -> dict[str, object]:
    folder = PurePosixPath(path).parts[0] if PurePosixPath(path).parts else ""
    slug = next((row["slug"] for row in _people_rows(report) if row["folder"] == folder), "")
    return {
        "path": path,
        "movedTo": moved_to,
        "already": already,
        "person": _person_counts(report, str(slug)),
        "summary": report.get("summary"),
    }


def _person_counts(report: dict[str, object], slug: str) -> dict[str, object]:
    for row in (report.get("summary") or {}).get("people") or []:
        if isinstance(row, dict) and str(row.get("slug")) == slug:
            return {
                "slug": slug,
                "files": int(row.get("files") or 0),
                "rejected": int(row.get("rejected") or 0),
            }
    return {"slug": slug, "files": 0, "rejected": 0}


def _free_name(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    if not target.exists():
        return target
    stem, suffix = os.path.splitext(name)
    number = 2
    while (folder / f"{stem} ({number}){suffix}").exists():
        number += 1
    return folder / f"{stem} ({number}){suffix}"


def _move(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(source, target)
    except PermissionError as exc:
        if getattr(exc, "winerror", None) == 32:
            raise ResultsError(
                409, "Dosya başka bir programda açık. Kapatıp yeniden deneyin."
            ) from exc
        raise ResultsError(
            500, "Dosya taşınamadı. Çıktı klasörüne yazılabildiğinden emin olun."
        ) from exc
    except OSError as exc:
        raise ResultsError(
            500, "Dosya taşınamadı. Çıktı klasörüne yazılabildiğinden emin olun."
        ) from exc


def _float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _box(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def is_video_path(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in VIDEO_EXTENSIONS
