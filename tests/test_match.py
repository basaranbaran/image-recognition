"""Matching policy: max over references, weak faces, no eye gate."""

from pathlib import Path

import numpy as np
import pytest

from recognize.match import (
    COSINE_THRESHOLD,
    DEFAULT_SAMPLE_SECONDS,
    Observation,
    ScoredPerson,
    decide_ranked,
    l2_normalize,
    max_similarity,
    needs_clothing_check,
    rank_people,
    summarize,
)


def test_max_reference_beats_averaged_vector() -> None:
    references = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)
    averaged = references.mean(axis=0)
    averaged = averaged / np.linalg.norm(averaged)
    best = max_similarity(query, references)
    averaged_score = float(averaged @ l2_normalize(query))
    assert best == pytest.approx(1.0)
    assert averaged_score < 0.9
    assert best > averaged_score

    ranked = rank_people(query, [("emre", "Emre", references, False)])
    kept = decide_ranked(ranked, threshold=0.9, margin=0.04, weak_band=0.08, clothing_matches=False)
    averaged_rows = l2_normalize(references.mean(axis=0)).reshape(1, -1)
    missed = decide_ranked(
        rank_people(query, [("emre", "Emre", averaged_rows, False)]),
        threshold=0.9,
        margin=0.04,
        weak_band=0.08,
        clothing_matches=False,
    )
    assert kept is not None and kept.score == pytest.approx(1.0)
    assert missed is None


def test_weak_score_accepted_only_with_matching_clothing() -> None:
    ranked = [ScoredPerson("emre", "Emre", 0.32, True)]
    decision = decide_ranked(
        ranked, threshold=0.363, margin=0.04, weak_band=0.08, clothing_matches=True
    )
    assert decision is not None
    assert decision.clothing_assisted is True
    assert decision.ambiguous is False


def test_weak_score_rejected_when_clothing_does_not_match() -> None:
    ranked = [ScoredPerson("emre", "Emre", 0.32, True)]
    assert (
        decide_ranked(ranked, threshold=0.363, margin=0.04, weak_band=0.08, clothing_matches=False)
        is None
    )


def test_person_without_hint_is_not_accepted_on_clothing() -> None:
    ranked = [ScoredPerson("ayse", "Ayşe", 0.32, False)]
    assert (
        decide_ranked(ranked, threshold=0.363, margin=0.04, weak_band=0.08, clothing_matches=True)
        is None
    )
    assert needs_clothing_check(ranked, threshold=0.363, margin=0.04, weak_band=0.08) is False


def test_clear_match_ignores_clothing() -> None:
    ranked = [ScoredPerson("emre", "Emre", 0.7, True)]
    decision = decide_ranked(
        ranked, threshold=0.363, margin=0.04, weak_band=0.08, clothing_matches=False
    )
    assert decision is not None
    assert decision.clothing_assisted is False
    assert needs_clothing_check(ranked, threshold=0.363, margin=0.04, weak_band=0.08) is False


def test_no_faces_match_nobody() -> None:
    hits, uncertain = summarize(
        [],
        kind="image",
        min_video_hits=2,
        strong_delta=0.12,
        threshold=0.363,
    )
    assert hits == []
    assert uncertain == []


def test_video_clothing_assist_needs_two_samples() -> None:
    one = [
        Observation(0, "emre", "Emre", 0.32, True, False),
    ]
    hits, _ = summarize(one, kind="video", min_video_hits=2, strong_delta=0.12, threshold=0.363)
    assert hits == []
    two = one + [Observation(1, "emre", "Emre", 0.33, True, False)]
    hits, _ = summarize(two, kind="video", min_video_hits=2, strong_delta=0.12, threshold=0.363)
    assert len(hits) == 1
    assert hits[0].clothing_assisted is True
    assert hits[0].samples == 2


def test_default_video_interval_is_half_a_second() -> None:
    assert DEFAULT_SAMPLE_SECONDS == 0.5


def test_crowd_scores_at_the_old_bar_are_rejected() -> None:
    """0.363 used to accept other people in this gallery (measured up to 0.441).

    A side view as hard as the weakest enrolled angle (0.604) still matches.
    """
    policy = dict(margin=0.04, weak_band=0.08, clothing_matches=False)
    impostor = [ScoredPerson("ibrahim-unyeli", "İbrahim Ünyeli", 0.441, False)]
    assert decide_ranked(impostor, threshold=0.363, **policy) is not None
    assert decide_ranked(impostor, threshold=COSINE_THRESHOLD, **policy) is None
    side_view = [ScoredPerson("ibrahim-unyeli", "İbrahim Ünyeli", 0.604, False)]
    kept = decide_ranked(side_view, threshold=COSINE_THRESHOLD, **policy)
    assert kept is not None
    assert kept.clothing_assisted is False
    assert 0.441 < COSINE_THRESHOLD <= 0.604


def test_matcher_has_no_eye_or_yaw_gate() -> None:
    root = Path(__file__).parents[1] / "src" / "recognize"
    for name in ("match.py", "engine.py", "clothing.py", "scan.py"):
        text = (root / name).read_text(encoding="utf-8").lower()
        for token in ("eye", "yaw", "blink", "göz"):
            assert token not in text, f"{name} contains {token}"


def _unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_second_look_uses_clear_event_faces_but_not_for_small_ones(tmp_path) -> None:
    from pathlib import Path

    from recognize.engine import GalleryRecognizer
    from recognize.gallery import Person
    from recognize.match import ScanOptions

    reference = _unit(1, 0, 0)
    clear = _unit(0.7, 0.714, 0)  # 0.70 to the webcam reference: an anchor
    profile = _unit(0.45, 0.85, 0.274)  # 0.45 to the webcam, 0.92 to the anchor
    person = Person("Baran", "baran", "", None, reference.reshape(1, -1), [], tmp_path)
    recognizer = GalleryRecognizer(None, [person], ScanOptions())
    recognizer._seen = [
        (Path("a.jpg"), 80.0, clear),
        (Path("b.jpg"), 40.0, profile),
        (Path("c.jpg"), 25.0, profile),
        (Path("d.jpg"), 40.0, _unit(0, 0, 1)),
    ]
    found = recognizer.event_matches()
    assert set(found) == {Path("a.jpg"), Path("b.jpg")}
    assert found[Path("b.jpg")][0].score > 0.9
    # The anchor's own photo is matched by the webcam reference, not by itself.
    assert found[Path("a.jpg")][0].score < 0.75

    recognizer._seen = [(Path("a.jpg"), 40.0, clear), (Path("b.jpg"), 40.0, profile)]
    assert Path("b.jpg") not in recognizer.event_matches()

    off = GalleryRecognizer(None, [person], ScanOptions(learn_from_event=False))
    off._seen = [(Path("a.jpg"), 80.0, clear), (Path("b.jpg"), 40.0, profile)]
    assert off.event_matches() == {}
