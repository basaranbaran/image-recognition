"""Copy (or move) event files into one folder per matched person."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from recognize.errors import AppError
from recognize.gallery import Person
from recognize.match import SMALL_FACE_PX, PersonHit, ScanOptions
from recognize.media import (
    FORMATS_LINE,
    VIDEO_EXTENSIONS,
    clone_file,
    collect_media,
    content_groups,
    directories_overlap,
    output_name,
)
from recognize.store import write_json_atomic
from recognize.textutil import folder_name

Progress = Callable[[int, int, str], None]

EMPTY_NOTE = "Dosya boş (0 bayt). Kopyalama yarım kalmış olabilir."
UNREADABLE_NOTE = "Videoda okunabilir kare yok. Kopyalama yarım kalmış olabilir."
# A partial report lands on disk after the first copy, then at this cadence.
CHECKPOINT_FILES = 25
CHECKPOINT_SECONDS = 10.0


class ScanCancelled(Exception):
    """The user stopped the scan. Not a file error."""


@dataclass
class ScanHooks:
    """Optional observers. The CLI passes none; the panel uses them for its progress view."""

    on_phase: Callable[[str], None] | None = None
    on_plan: Callable[[list[Path]], None] | None = None
    on_file: Callable[[dict[str, object], float], None] | None = None
    on_frame: Callable[[], None] | None = None
    # Called once with the output folder before the first report write of this scan.
    archive: Callable[[Path], object] | None = None


@dataclass
class InspectionLike:
    hits: list[PersonHit]
    uncertain: list[PersonHit]


class Recognizer:
    def inspect_image(
        self,
        path: Path,
        progress: Callable[[str], None] | None = None,
    ) -> InspectionLike:
        raise NotImplementedError

    def inspect_video(
        self,
        path: Path,
        progress: Callable[[str], None] | None = None,
    ) -> InspectionLike:
        raise NotImplementedError


def run_scan(
    source: Path,
    dest: Path,
    recognizer: Recognizer,
    people: list[Person],
    options: ScanOptions,
    on_progress: Progress | None = None,
    *,
    should_stop: Callable[[], bool] | None = None,
    hooks: ScanHooks | None = None,
) -> dict[str, object]:
    """Inspect every file and copy the matches. Raises ``ScanCancelled`` only when the stop
    came before any file was finished; after that a partial report is written and returned.
    """
    source = source.expanduser().resolve()
    dest = dest.expanduser().resolve()
    validate_scan(source, dest, people)
    hooks = hooks or ScanHooks()
    stop = should_stop or _never
    files = collect_media(source)
    if not files:
        raise AppError(f"Klasörde fotoğraf ya da video yok ({FORMATS_LINE}).")
    _call(hooks.on_plan, files)
    dest.mkdir(parents=True, exist_ok=True)
    _call(hooks.on_phase, "hash")
    _progress(on_progress, 0, len(files), "Aynı dosyanın kopyaları aranıyor")
    copies = content_groups(files, stop)
    if stop():
        raise ScanCancelled
    wanted = {person.slug for person in people}
    writer = _ReportWriter(dest, hooks)

    _call(hooks.on_phase, "files")
    results: list[dict[str, object]] = []
    by_path: dict[Path, dict[str, object]] = {}
    total = len(files)
    stopped = False
    for index, path in enumerate(files, start=1):
        if stop():
            stopped = True
            break
        label = path.relative_to(source).as_posix()
        _progress(on_progress, index - 1, total, label)

        def label_progress(text: str, current: str = label) -> None:
            if stop():
                raise ScanCancelled
            _call(hooks.on_frame)
            _progress(on_progress, index - 1, total, text or current)

        started = time.monotonic()
        original = copies.get(path)
        try:
            if original is not None and original in by_path:
                record = _copy_record(path, original, source, by_path[original], options)
            else:
                record = _inspect_one(
                    path, source, dest, recognizer, options, label_progress, wanted
                )
        except ScanCancelled:
            stopped = True
            break
        by_path[path] = record
        results.append(record)
        _call(hooks.on_file, record, time.monotonic() - started)
        if record["copiedTo"]:
            writer.checkpoint(
                lambda: _report(
                    source, dest, people, options, results, done=len(results), total=total
                )
            )

    if stopped:
        if not results:
            raise ScanCancelled
        report = _report(source, dest, people, options, results, done=len(results), total=total)
        report["stoppedAt"] = datetime.now(UTC).isoformat()
        _call(hooks.on_phase, "writing")
        writer.write(report)
        return report

    learn = getattr(recognizer, "event_matches", None)
    if callable(learn):
        _call(hooks.on_phase, "secondLook")
        _progress(on_progress, total, total, "Etkinlikteki net karelerden ikinci bakış")
        _add_learned(learn(), source, dest, options, wanted, results)

    _call(hooks.on_phase, "writing")
    _progress(on_progress, total, total, "")
    report = _report(source, dest, people, options, results)
    writer.write(report)
    return report


def validate_scan(source: Path, dest: Path, people: list[Person]) -> None:
    source = source.expanduser().resolve()
    dest = dest.expanduser().resolve()
    if not people:
        raise AppError("Henüz kimse tanıtılmadı. Önce tanıtım fotoğrafları ekleyin.")
    if not source.is_dir():
        raise AppError("Kaynak klasör bulunamadı.")
    if directories_overlap(source, dest):
        raise AppError("Kaynak ve çıktı klasörü iç içe olamaz. Çıktıyı başka bir klasöre verin.")


class _ReportWriter:
    """Archives the previous report once, then writes checkpoints at a bounded cadence."""

    def __init__(self, dest: Path, hooks: ScanHooks) -> None:
        self._dest = dest
        self._hooks = hooks
        self._archived = False
        self._since_write = 0
        self._last_write = 0.0
        self._written_once = False

    def checkpoint(self, build: Callable[[], dict[str, object]]) -> None:
        self._since_write += 1
        now = time.monotonic()
        due = (
            not self._written_once
            or self._since_write >= CHECKPOINT_FILES
            or now - self._last_write >= CHECKPOINT_SECONDS
        )
        if not due:
            return
        self.write(build())

    def write(self, report: dict[str, object]) -> None:
        if not self._archived:
            self._archived = True
            try:
                _call(self._hooks.archive, self._dest)
            except OSError:
                logging.exception("Could not archive the previous report")
        write_json_atomic(self._dest / "rapor.json", report)
        self._written_once = True
        self._since_write = 0
        self._last_write = time.monotonic()


def _inspect_one(
    path: Path,
    source: Path,
    dest: Path,
    recognizer: Recognizer,
    options: ScanOptions,
    progress: Callable[[str], None],
    wanted: set[str] | None = None,
) -> dict[str, object]:
    kind = "video" if path.suffix.lower() in VIDEO_EXTENSIONS else "image"
    relative = path.relative_to(source).as_posix()
    try:
        empty = path.stat().st_size == 0
    except OSError:
        return _error_record(relative, kind, "Dosya okunamadı.")
    if empty:
        record = _error_record(relative, kind, None)
        record["skipped"] = EMPTY_NOTE
        return record
    try:
        if kind == "video":
            inspection = recognizer.inspect_video(path, progress)
        else:
            inspection = recognizer.inspect_image(path, progress)
    except ScanCancelled:
        raise
    except AppError as exc:
        if _is_unreadable(exc):
            record = _error_record(relative, kind, None)
            record["skipped"] = UNREADABLE_NOTE
            return record
        return _error_record(relative, kind, str(exc))
    except Exception:
        logging.exception("Could not read %s", path)
        return _error_record(relative, kind, "Bu dosya okunamadı.")

    if wanted is not None:
        # The recognizer ranks every enrolled face; unselected people get no folder.
        inspection = InspectionLike(
            [hit for hit in inspection.hits if hit.slug in wanted],
            [hit for hit in inspection.uncertain if hit.slug in wanted],
        )
    copied: list[str] = []
    try:
        for hit in inspection.hits:
            target_dir = dest / folder_name(hit.name)
            target = _place(path, target_dir, output_name(path, source))
            copied.append(target.relative_to(dest).as_posix())
        moved = False
        if options.move and copied:
            path.unlink()
            moved = True
    except OSError:
        logging.exception("Could not copy %s", path)
        return _error_record(relative, kind, "Dosya klasöre yazılamadı.")

    return {
        "source": relative,
        "kind": kind,
        "matches": [_hit_dict(hit) for hit in inspection.hits],
        "uncertain": [_hit_dict(hit) for hit in inspection.uncertain],
        "copiedTo": copied,
        "moved": moved,
        "error": None,
        "skipped": None,
        "duplicateOf": None,
    }


def _is_unreadable(exc: AppError) -> bool:
    # Imported lazily: engine pulls in OpenCV models, which the fake recognizers never need.
    from recognize.engine import UnreadableMedia

    return isinstance(exc, UnreadableMedia)


def _add_learned(
    found: dict[Path, list[PersonHit]],
    source: Path,
    dest: Path,
    options: ScanOptions,
    wanted: set[str],
    results: list[dict[str, object]],
) -> None:
    """Write files the second look found; their matches are marked ``learned``."""
    by_source = {record["source"]: record for record in results}
    for path, hits in found.items():
        relative = path.relative_to(source).as_posix()
        record = by_source.get(relative)
        if record is None or record["error"] or record.get("duplicateOf"):
            continue
        known = {match["slug"] for match in record["matches"]}
        fresh = [hit for hit in hits if hit.slug in wanted and hit.slug not in known]
        if not fresh:
            continue
        # A moved file is gone from the source; its first copy stands in for it.
        origin = path if path.exists() else dest / str(record["copiedTo"][0])
        try:
            for hit in fresh:
                target = _place(origin, dest / folder_name(hit.name), output_name(path, source))
                record["copiedTo"].append(target.relative_to(dest).as_posix())
                record["matches"].append({**_hit_dict(hit), "learned": True})
            slugs = {hit.slug for hit in fresh}
            record["uncertain"] = [row for row in record["uncertain"] if row["slug"] not in slugs]
            if options.move and not record["moved"] and path.exists():
                path.unlink()
                record["moved"] = True
        except OSError:
            logging.exception("Could not copy %s", path)
            record["error"] = "Dosya klasöre yazılamadı."
            continue
        for copy in results:
            if copy.get("duplicateOf") != relative:
                continue
            copy["matches"] = list(record["matches"])
            copy["uncertain"] = list(record["uncertain"])
            twin = source / str(copy["source"])
            if options.move and record["moved"] and not copy["moved"] and twin.exists():
                twin.unlink()
                copy["moved"] = True


def _copy_record(
    path: Path,
    original: Path,
    source: Path,
    first: dict[str, object],
    options: ScanOptions,
) -> dict[str, object]:
    """A byte-identical copy reuses the first file's result and is not written twice."""
    relative = path.relative_to(source).as_posix()
    record = dict(first)
    record.update(
        source=relative,
        copiedTo=[],
        moved=False,
        duplicateOf=original.relative_to(source).as_posix(),
    )
    if options.move and first.get("moved"):
        try:
            path.unlink()
            record["moved"] = True
        except OSError:
            logging.exception("Could not remove copy %s", path)
            record["error"] = "Kopya dosya kaynaktan silinemedi."
    return record


def _place(source: Path, dest_dir: Path, name: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / name
    if target.resolve() == source.resolve():
        raise AppError("Kaynak ile çıktı aynı dosya.")
    # Resolve first: a symlinked photo becomes a real file under its own name, never a link.
    clone_file(source.resolve(), target)
    return target


def _hit_dict(hit: PersonHit) -> dict[str, object]:
    return {
        "name": hit.name,
        "slug": hit.slug,
        "score": round(float(hit.score), 4),
        "samples": hit.samples,
        "clothingAssisted": hit.clothing_assisted,
        "facePx": round(float(hit.face_px), 1),
        "smallFace": 0 < hit.face_px < SMALL_FACE_PX,
        "learned": False,
        "box": None if hit.box is None else [round(float(value), 4) for value in hit.box],
    }


def _error_record(relative: str, kind: str, message: str | None) -> dict[str, object]:
    return {
        "source": relative,
        "kind": kind,
        "matches": [],
        "uncertain": [],
        "copiedTo": [],
        "moved": False,
        "error": message,
        "skipped": None,
        "duplicateOf": None,
    }


def summarize_results(
    results: list[dict[str, object]],
    people_rows: list[dict[str, object]],
) -> dict[str, object]:
    """The report's ``summary`` block, counted from its ``results``.

    ``people_rows`` carry ``name``, ``slug`` and ``clothingLabel``; a correction made
    later re-counts with the same function, so the two can never drift apart.
    """
    counts: dict[str, dict[str, object]] = {
        str(row["slug"]): {
            "name": row["name"],
            "slug": row["slug"],
            "files": 0,
            "clothingAssisted": 0,
            "smallFaces": 0,
            "learned": 0,
            "rejected": 0,
            "clothingLabel": row.get("clothingLabel"),
        }
        for row in people_rows
    }
    errors = 0
    matched = 0
    images = 0
    videos = 0
    empty = 0
    unreadable = 0
    copies = 0
    rejected = 0
    for record in results:
        if record["kind"] == "video":
            videos += 1
        else:
            images += 1
        if record["error"]:
            errors += 1
        skipped = record.get("skipped")
        if skipped == UNREADABLE_NOTE:
            unreadable += 1
        elif skipped:
            empty += 1
        if record.get("duplicateOf"):
            copies += 1
            continue
        for row in record.get("rejected") or []:
            if isinstance(row, dict):
                rejected += 1
                bucket = counts.get(str(row.get("slug")))
                if bucket is not None:
                    bucket["rejected"] = int(bucket["rejected"]) + 1
        matches = record["matches"]
        if isinstance(matches, list) and matches:
            matched += 1
            for hit in matches:
                if not isinstance(hit, dict):
                    continue
                bucket = counts.get(str(hit["slug"]))
                if bucket is None:
                    continue
                bucket["files"] = int(bucket["files"]) + 1
                if hit.get("clothingAssisted"):
                    bucket["clothingAssisted"] = int(bucket["clothingAssisted"]) + 1
                if hit.get("smallFace"):
                    bucket["smallFaces"] = int(bucket["smallFaces"]) + 1
                if hit.get("learned"):
                    bucket["learned"] = int(bucket["learned"]) + 1
    return {
        "files": len(results),
        "images": images,
        "videos": videos,
        "matchedFiles": matched,
        "errors": errors,
        "emptyFiles": empty,
        "unreadable": unreadable,
        "duplicates": copies,
        "rejected": rejected,
        "people": list(counts.values()),
    }


def _report(
    source: Path,
    dest: Path,
    people: list[Person],
    options: ScanOptions,
    results: list[dict[str, object]],
    *,
    done: int | None = None,
    total: int | None = None,
) -> dict[str, object]:
    rows = [
        {
            "name": person.name,
            "slug": person.slug,
            "clothingLabel": None if person.clothing is None else person.clothing.label,
        }
        for person in people
    ]
    partial = done is not None and total is not None
    return {
        "scannedAt": datetime.now(UTC).isoformat(),
        "source": str(source),
        "destination": str(dest),
        "threshold": options.threshold,
        "sampleSeconds": options.sample_seconds,
        "move": options.move,
        "complete": not partial,
        "progress": {"done": done, "total": total} if partial else None,
        "people": [
            {
                "name": person.name,
                "slug": person.slug,
                "clothing": None if person.clothing is None else person.clothing.name,
            }
            for person in people
        ],
        "results": results,
        "summary": summarize_results(results, rows),
    }


def _progress(callback: Progress | None, done: int, total: int, label: str) -> None:
    if callback is not None:
        callback(done, total, label)


def _call(callback: Callable[..., object] | None, *args: object) -> None:
    if callback is not None:
        callback(*args)


def _never() -> bool:
    return False
