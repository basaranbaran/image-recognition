"""Shared enrollment rules for the CLI and the panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from recognize.clothing import ClothingHint
from recognize.errors import AppError
from recognize.gallery import Person, add_references
from recognize.messages import SINGLE_REFERENCE_WARNING


def enroll_faces(
    name: str,
    samples: list[tuple[np.ndarray, float, bytes, str]],
    clothing: ClothingHint | None,
    *,
    update_clothing: bool,
    root: Path | None = None,
) -> tuple[Person, str | None]:
    if not samples:
        raise AppError("Seçilen fotoğraflarda kullanılacak yüz yok.")
    person, _created = add_references(
        name,
        samples,
        clothing,
        update_clothing=update_clothing,
        root=root,
    )
    warning = SINGLE_REFERENCE_WARNING if len(person.references) == 1 else None
    return person, warning
