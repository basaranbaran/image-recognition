"""Shared fixtures: a temporary app home and a scanned output built with the real run_scan."""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from recognize.gallery import Person
from recognize.match import PersonHit, ScanOptions
from recognize.paths import people_dir
from recognize.scan import InspectionLike, run_scan


def jpeg_bytes(
    width: int = 64, height: int = 48, orientation: int | None = 6, shade: int = 200
) -> bytes:
    image = Image.new("RGB", (width, height), (shade, 30, 30))
    exif = Image.Exif()
    if orientation:
        exif[0x0112] = orientation  # 6 = shown rotated 90° clockwise
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", exif=exif)
    return buffer.getvalue()


class FakeRecognizer:
    def __init__(self, hits: dict[str, list[PersonHit]]) -> None:
        self._hits = hits
        self.seen: list[str] = []

    def inspect_image(self, path: Path, progress=None) -> InspectionLike:
        del progress
        self.seen.append(path.name)
        return InspectionLike(self._hits.get(path.name, []), [])

    def inspect_video(self, path: Path, progress=None) -> InspectionLike:
        del progress
        self.seen.append(path.name)
        return InspectionLike(self._hits.get(path.name, []), [])


def person(name: str, slug: str, root: Path | None = None) -> Person:
    return Person(
        name=name,
        slug=slug,
        created_at="",
        clothing=None,
        embeddings=np.ones((1, 4), dtype=np.float32),
        references=[],
        root=root or Path("."),
    )


def scanned_output(tmp_path: Path, *, move: bool = False) -> Path:
    """Two people, two photos; ``ikisi.jpg`` holds both, with Emre's face small."""
    source = tmp_path / "etkinlik"
    source.mkdir(exist_ok=True)
    (source / "ikisi.jpg").write_bytes(jpeg_bytes())
    (source / "emre.jpg").write_bytes(jpeg_bytes(80, 60, None, 120))
    hits = {
        "ikisi.jpg": [
            PersonHit("ali", "Ali Veli", 0.71, 1, False, 90.0, (0.1, 0.1, 0.2, 0.3)),
            PersonHit("emre", "Emre Kaya", 0.52, 1, False, 25.0),
        ],
        "emre.jpg": [PersonHit("emre", "Emre Kaya", 0.66, 1, False, 70.0)],
    }
    output = tmp_path / "cikti"
    people = [person("Ali Veli", "ali"), person("Emre Kaya", "emre")]
    run_scan(source, output, FakeRecognizer(hits), people, ScanOptions(move=move))
    if os.environ.get("RECOGNIZE_HOME"):
        # The panel shows only people whose gallery folder exists; these two are still enrolled.
        for enrolled in people:
            (people_dir() / enrolled.slug).mkdir(parents=True, exist_ok=True)
    return output


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "home"
    root.mkdir()
    monkeypatch.setenv("RECOGNIZE_HOME", str(root))
    return root
