"""Enrolled people: one embedding row per reference photo, plus an optional hue hint."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from recognize.clothing import ClothingHint
from recognize.errors import AppError
from recognize.match import l2_normalize
from recognize.paths import data_dir, people_dir
from recognize.textutil import clean_person_name, slugify, tr_lower

_LOCK = threading.Lock()
_MARKER = ".silinen.json"
# One camera sweep stores 16 views. 48 let a person finish only two sweeps; the
# third was refused after the user had already turned their head the whole way.
MAX_REFERENCES = 96


@dataclass(frozen=True)
class ReferenceSample:
    """One stored face. Extra fields describe the webcam angle; they are not averaged."""

    embedding: np.ndarray
    det_score: float
    jpeg: bytes
    source_name: str
    yaw: float | None = None
    pitch: float | None = None
    landmarks: tuple[float, ...] | None = None


@dataclass
class Person:
    name: str
    slug: str
    created_at: str
    clothing: ClothingHint | None
    embeddings: np.ndarray
    references: list[dict[str, object]]
    root: Path

    @property
    def directory(self) -> Path:
        return self.root / self.slug


def load_people(root: Path | None = None) -> list[Person]:
    base = root if root is not None else people_dir()
    if not base.exists():
        return []
    people = [
        _read_person(entry, base)
        for entry in base.iterdir()
        if entry.is_dir() and (entry / "person.json").is_file()
    ]
    people.sort(key=lambda person: tr_lower(person.name))
    return people


def load_people_safe(root: Path | None = None) -> tuple[list[Person], list[dict[str, str]]]:
    """Like ``load_people`` but one broken folder does not hide everyone else.

    Returns the readable people and a list of ``{"folder", "detail"}`` problems.
    """
    base = root if root is not None else people_dir()
    if not base.exists():
        return [], []
    people: list[Person] = []
    problems: list[dict[str, str]] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if not (entry / "person.json").is_file():
            if any(entry.iterdir()):
                problems.append(
                    {
                        "folder": entry.name,
                        "detail": f"{entry.name} klasöründe kayıt dosyası yok. "
                        "Kaydı silip yeniden tanıtın.",
                    }
                )
            continue
        try:
            people.append(_read_person(entry, base))
        except (AppError, OSError, ValueError, KeyError) as exc:
            reason = str(exc) if isinstance(exc, AppError) else f"{entry.name} kaydı okunamadı."
            problems.append(
                {"folder": entry.name, "detail": f"{reason} Kaydı silip yeniden tanıtın."}
            )
    people.sort(key=lambda person: tr_lower(person.name))
    return people, problems


def find_person(name_or_slug: str, root: Path | None = None) -> Person | None:
    raw = name_or_slug.strip()
    if not raw:
        return None
    needle_slug = raw.lower()
    try:
        needle_name = tr_lower(clean_person_name(raw))
    except AppError:
        needle_name = tr_lower(raw)
    for person in load_people(root):
        if person.slug == needle_slug or tr_lower(person.name) == needle_name:
            return person
    return None


def reference_room(name_or_slug: str, root: Path | None = None) -> int:
    """How many more reference rows this person can take."""
    person = find_person(name_or_slug, root)
    used = 0 if person is None else len(person.references)
    return max(0, MAX_REFERENCES - used)


def add_references(
    name: str,
    samples: list[tuple[np.ndarray, float, bytes, str] | ReferenceSample],
    clothing: ClothingHint | None,
    *,
    update_clothing: bool,
    root: Path | None = None,
) -> tuple[Person, bool]:
    """Append reference rows. Returns the person and whether the record is new.

    Each sample is ``(embedding, det_score, jpeg_bytes, source_name)`` or a
    ``ReferenceSample``. Rows stay separate; nothing is averaged.
    """
    if not samples:
        raise AppError("Eklenecek yüz yok.")
    cleaned = clean_person_name(name)
    with _LOCK:
        base = root if root is not None else people_dir()
        base.mkdir(parents=True, exist_ok=True)
        existing = find_person(cleaned, base)
        if existing is None:
            slug = _unique_slug(cleaned, base)
            created_at = datetime.now(UTC).isoformat()
            prior_rows: np.ndarray | None = None
            prior_refs: list[dict[str, object]] = []
            stored_clothing = clothing if update_clothing else None
            created = True
        else:
            slug = existing.slug
            created_at = existing.created_at
            prior_rows = existing.embeddings
            prior_refs = list(existing.references)
            stored_clothing = clothing if update_clothing else existing.clothing
            created = False

        if len(prior_refs) + len(samples) > MAX_REFERENCES:
            raise AppError(f"Bir kişiye en fazla {MAX_REFERENCES} referans eklenebilir.")

        parsed = [_sample_parts(sample) for sample in samples]
        new_rows = np.stack([l2_normalize(row) for row, _score, _jpeg, _name, _meta in parsed])
        if prior_rows is None or prior_rows.size == 0:
            rows = new_rows
        else:
            rows = np.vstack([prior_rows, new_rows])

        directory = base / slug
        ref_dir = directory / "referans"
        ref_dir.mkdir(parents=True, exist_ok=True)
        refs = list(prior_refs)
        start = len(refs) + 1
        for offset, (_embedding, det_score, jpeg, source_name, meta) in enumerate(parsed):
            filename = f"{start + offset:02d}.jpg"
            (ref_dir / filename).write_bytes(jpeg)
            row: dict[str, object] = {
                "file": f"referans/{filename}",
                "detScore": round(float(det_score), 4),
                "sourceName": source_name,
            }
            row.update(meta)
            refs.append(row)

        person = Person(
            name=cleaned,
            slug=slug,
            created_at=created_at,
            clothing=stored_clothing,
            embeddings=rows.astype(np.float32),
            references=refs,
            root=base,
        )
        _write_person(person)
        return person, created


def _sample_parts(
    sample: tuple[np.ndarray, float, bytes, str] | ReferenceSample,
) -> tuple[np.ndarray, float, bytes, str, dict[str, object]]:
    if isinstance(sample, ReferenceSample):
        meta: dict[str, object] = {}
        if sample.yaw is not None:
            meta["yaw"] = round(float(sample.yaw), 4)
        if sample.pitch is not None:
            meta["pitch"] = round(float(sample.pitch), 4)
        if sample.landmarks:
            meta["landmarks"] = [round(float(value), 2) for value in sample.landmarks]
        return (
            sample.embedding,
            float(sample.det_score),
            sample.jpeg,
            sample.source_name,
            meta,
        )
    embedding, det_score, jpeg, source_name = sample
    return embedding, float(det_score), jpeg, source_name, {}


def delete_person(name_or_slug: str, root: Path | None = None) -> Person:
    with _LOCK:
        base = root if root is not None else people_dir()
        person = find_person(name_or_slug, base)
        if person is None:
            raise AppError("Böyle bir kişi yok.")
        shutil.rmtree(person.directory)
        return person


@dataclass(frozen=True)
class Removal:
    """What a soft delete moved aside, so it can be put back."""

    slug: str
    name: str
    token: str


def present_slugs(root: Path | None = None) -> set[str]:
    """Folder names of everyone still in the gallery, broken folders included.

    A person deleted in the panel has left this folder, and the results stop showing them.
    """
    base = root if root is not None else people_dir()
    if not base.is_dir():
        return set()
    return {
        entry.name
        for entry in base.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    }


def soft_delete(slug: str, root: Path | None = None, trash: Path | None = None) -> Removal:
    """Move ``<root>/<slug>`` to ``<trash>/<slug>-<stamp>``. Works for a broken folder too."""
    with _LOCK:
        base = root if root is not None else people_dir()
        bin_dir = trash if trash is not None else _trash_dir()
        directory = base / slug
        if not directory.is_dir():
            raise AppError("Böyle bir kişi yok.")
        name = slug
        try:
            payload = json.loads((directory / "person.json").read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("name"), str):
                name = payload["name"]
        except (OSError, ValueError):
            pass
        bin_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        token = f"{slug}-{stamp}"
        number = 2
        while (bin_dir / token).exists():
            token = f"{slug}-{stamp}-{number}"
            number += 1
        shutil.move(str(directory), str(bin_dir / token))
        marker = {"slug": slug, "name": name, "deletedAt": datetime.now(UTC).isoformat()}
        (bin_dir / token / _MARKER).write_text(
            json.dumps(marker, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return Removal(slug=slug, name=name, token=token)


def restore_person(token: str, root: Path | None = None, trash: Path | None = None) -> Removal:
    """Put a soft-deleted folder back under its slug. Refuses when that slug is taken again."""
    with _LOCK:
        base = root if root is not None else people_dir()
        bin_dir = trash if trash is not None else _trash_dir()
        source = bin_dir / token
        if not source.is_dir():
            raise AppError("Geri alınacak bir silme yok.")
        marker: dict[str, object] = {}
        try:
            loaded = json.loads((source / _MARKER).read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                marker = loaded
        except (OSError, ValueError):
            pass
        slug = str(marker.get("slug") or token.rsplit("-", 1)[0])
        name = str(marker.get("name") or slug)
        target = base / slug
        if target.exists():
            raise AppError("Bu adla yeni bir kayıt var; geri alınamadı.")
        base.mkdir(parents=True, exist_ok=True)
        (source / _MARKER).unlink(missing_ok=True)
        shutil.move(str(source), str(target))
        return Removal(slug=slug, name=name, token=token)


def purge_deleted(days: int = 7, trash: Path | None = None) -> int:
    """Remove soft-deleted folders older than ``days``. Returns how many went."""
    bin_dir = trash if trash is not None else _trash_dir()
    if not bin_dir.is_dir():
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    for entry in bin_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            if entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry)
                removed += 1
        except OSError:
            continue
    return removed


def _trash_dir() -> Path:
    return data_dir() / "silinenler"


def _unique_slug(name: str, root: Path) -> str:
    base = slugify(name)
    used = {entry.name for entry in root.iterdir() if entry.is_dir()}
    if base not in used:
        return base
    number = 2
    while f"{base}-{number}" in used:
        number += 1
    return f"{base}-{number}"


def _read_person(directory: Path, root: Path) -> Person:
    payload = json.loads((directory / "person.json").read_text(encoding="utf-8"))
    clothing_payload = payload.get("clothing")
    clothing = None
    if isinstance(clothing_payload, dict):
        clothing = ClothingHint.from_dict(clothing_payload)
    embeddings_path = directory / "embeddings.npy"
    if not embeddings_path.is_file():
        raise AppError(f"{directory.name} kaydında yüz vektörü yok.")
    embeddings = np.load(embeddings_path).astype(np.float32)
    references = list(payload.get("references") or [])
    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or not references:
        raise AppError(f"{directory.name} kaydı bozuk.")
    if embeddings.shape[0] != len(references):
        # Both files only ever grow by appending, so the common prefix is consistent.
        keep = min(embeddings.shape[0], len(references))
        logging.getLogger("recognize").warning(
            "%s: %d vektör, %d referans; ilk %d satır kullanılıyor",
            directory.name,
            embeddings.shape[0],
            len(references),
            keep,
        )
        embeddings = embeddings[:keep]
        references = references[:keep]
    return Person(
        name=str(payload["name"]),
        slug=str(payload["slug"]),
        created_at=str(payload.get("createdAt") or ""),
        clothing=clothing,
        embeddings=embeddings,
        references=references,
        root=root,
    )


def _write_person(person: Person) -> None:
    directory = person.directory
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": person.name,
        "slug": person.slug,
        "createdAt": person.created_at,
        "clothing": None if person.clothing is None else person.clothing.to_dict(),
        "references": person.references,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    temporary = directory / "person.json.tmp"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(directory / "person.json")
    vectors = directory / "embeddings.tmp.npy"
    np.save(vectors, person.embeddings.astype(np.float32))
    vectors.replace(directory / "embeddings.npy")
