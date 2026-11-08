"""Gallery keeps every reference row and the hue range."""

from pathlib import Path

import numpy as np

from recognize.clothing import hint_for
from recognize.gallery import add_references, load_people


def test_two_references_stay_separate_and_yellow_hue_is_stored(tmp_path: Path) -> None:
    first = np.array([1.0, 0.0], dtype=np.float32)
    second = np.array([0.0, 1.0], dtype=np.float32)
    hint = hint_for("sarı")
    person, created = add_references(
        "Emre",
        [
            (first, 0.9, b"jpeg-a", "on.jpg"),
            (second, 0.8, b"jpeg-b", "yan.jpg"),
        ],
        hint,
        update_clothing=True,
        root=tmp_path,
    )
    assert created is True
    assert person.embeddings.shape == (2, 2)
    average = person.embeddings.mean(axis=0)
    assert not np.allclose(person.embeddings[0], average)

    loaded = load_people(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].embeddings.shape == (2, 2)
    assert loaded[0].clothing is not None
    assert loaded[0].clothing.hue_min == 10
    assert loaded[0].clothing.hue_max == 25
    assert loaded[0].clothing.label == "sarı"
    payload = (tmp_path / loaded[0].slug / "person.json").read_text(encoding="utf-8")
    assert '"hueMin": 10' in payload
    assert '"hueMax": 25' in payload

    again, created_again = add_references(
        "emre",
        [(np.array([0.0, -1.0], dtype=np.float32), 0.7, b"jpeg-c", "goz.jpg")],
        None,
        update_clothing=False,
        root=tmp_path,
    )
    assert created_again is False
    assert again.embeddings.shape == (3, 2)
    assert again.clothing is not None
    assert again.clothing.name == "sari"
