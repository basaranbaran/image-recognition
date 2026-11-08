"""The event folder is discovered without touching code or output directories."""

from pathlib import Path

import pytest

from recognize.media import directories_overlap
from recognize.paths import default_output_dir, default_pool_dir, event_source_dir


def test_event_folder_skips_code_model_and_output_dirs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RECOGNIZE_HOME", str(tmp_path))
    event = tmp_path / "etkinlik"
    (event / "alt").mkdir(parents=True)
    (event / "alt" / "a.JPG").write_bytes(b"a")
    (event / "b.mp4").write_bytes(b"b")
    for name in ("src", "tests", "modeller", "veri", "cikti", ".venv"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "x.jpg").write_bytes(b"x")
    found = event_source_dir()
    assert found == event.resolve()
    assert not directories_overlap(found, default_output_dir())


def test_real_event_folder_is_not_the_output_dir() -> None:
    event = event_source_dir(prefer_pool=False)
    if event is None:
        pytest.skip("Etkinlik fotoğrafları bu kopyada yok (depoya girmez).")
    assert event.name not in {"src", "tests", "modeller", "veri", "cikti", ".venv"}
    assert "hackathon" in event.name
    source = event_source_dir()
    assert source is not None
    # Once the unique pool exists it is the default source; either way it is not the output.
    assert source in {event, default_pool_dir().resolve()}
    assert not directories_overlap(source, default_output_dir())
