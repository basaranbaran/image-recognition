"""Face decisions. Every reference is kept; the score is the best cosine, not an average."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 0.363 is OpenCV's one-to-one cosine default. Scores here are the best of every
# reference, and a crowd photo holds many faces. On this gallery those other
# people landed at 0.363–0.441. The enrolled angles still match each other at
# 0.604 and above, so 0.50 drops the impostor band and keeps a hard side view.
COSINE_THRESHOLD = 0.5
MARGIN = 0.04
WEAK_BAND = 0.08
STRONG_DELTA = 0.12
MIN_VIDEO_HITS = 2
DEFAULT_SAMPLE_SECONDS = 0.5


@dataclass(frozen=True)
class ScanOptions:
    threshold: float = COSINE_THRESHOLD
    margin: float = MARGIN
    weak_band: float = WEAK_BAND
    strong_delta: float = STRONG_DELTA
    min_video_hits: int = MIN_VIDEO_HITS
    sample_seconds: float = DEFAULT_SAMPLE_SECONDS
    image_max_edge: int = 1920
    video_max_edge: int = 1280
    min_det_score: float = 0.5
    min_face_px: int = 24
    move: bool = False
    learn_from_event: bool = True


@dataclass(frozen=True)
class ScoredPerson:
    slug: str
    name: str
    score: float
    has_clothing_hint: bool


@dataclass(frozen=True)
class FaceDecision:
    slug: str
    name: str
    score: float
    clothing_assisted: bool
    ambiguous: bool


@dataclass(frozen=True)
class Observation:
    frame: int
    slug: str
    name: str
    score: float
    clothing_assisted: bool
    ambiguous: bool
    face_px: float = 0.0
    # Face box as fractions of the upright frame: (x, y, width, height), each in 0..1.
    box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class PersonHit:
    slug: str
    name: str
    score: float
    samples: int
    clothing_assisted: bool
    # Shorter side of the best-scoring face. Below SMALL_FACE_PX a look-alike can pass.
    face_px: float = 0.0
    box: tuple[float, float, float, float] | None = None


# On this event, both wrong matches at 0.50 were 24–25 px faces of one look-alike.
SMALL_FACE_PX = 32

# Second look: a face matched this clearly shows that day's light, hair and camera,
# so it serves as an extra reference for the rest of the scan. On this event the 20
# such faces were all the right person; learned matches below SMALL_FACE_PX were all
# the look-alike, so those stay off.
ANCHOR_SCORE = 0.60
ANCHOR_MIN_PX = 48


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flat))
    if norm < 1e-12:
        raise ValueError("embedding norm is zero")
    return flat / norm


def max_similarity(embedding: np.ndarray, references: np.ndarray) -> float:
    """Highest cosine against every reference row."""
    query = l2_normalize(embedding)
    refs = np.asarray(references, dtype=np.float32)
    if refs.ndim == 1:
        refs = refs.reshape(1, -1)
    norms = np.linalg.norm(refs, axis=1, keepdims=True)
    refs = refs / np.clip(norms, 1e-12, None)
    return float(np.max(refs @ query))


def rank_people(
    embedding: np.ndarray,
    people: list[tuple[str, str, np.ndarray, bool]],
) -> list[ScoredPerson]:
    scored = [
        ScoredPerson(slug, name, max_similarity(embedding, rows), has_hint)
        for slug, name, rows, has_hint in people
        if rows.size
    ]
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored


def needs_clothing_check(
    ranked: list[ScoredPerson],
    *,
    threshold: float,
    margin: float,
    weak_band: float,
) -> bool:
    """Clothing is consulted only for a weak face of someone who has a hint."""
    if not ranked:
        return False
    best = ranked[0]
    if not best.has_clothing_hint:
        return False
    low = threshold - weak_band
    if best.score < low or best.score >= threshold:
        return False
    if len(ranked) > 1:
        second = ranked[1].score
        if second >= low and best.score - second < margin:
            return False
    return True


def decide_ranked(
    ranked: list[ScoredPerson],
    *,
    threshold: float,
    margin: float,
    weak_band: float,
    clothing_matches: bool,
) -> FaceDecision | None:
    if not ranked:
        return None
    best = ranked[0]
    second = ranked[1].score if len(ranked) > 1 else None
    low = threshold - weak_band

    if best.score >= threshold:
        ambiguous = second is not None and second >= threshold and best.score - second < margin
        return FaceDecision(best.slug, best.name, best.score, False, ambiguous)

    if best.score < low:
        return None
    if not best.has_clothing_hint or not clothing_matches:
        return None
    if second is not None and second >= low and best.score - second < margin:
        return None
    return FaceDecision(best.slug, best.name, best.score, True, False)


def summarize(
    observations: list[Observation],
    *,
    kind: str,
    min_video_hits: int,
    strong_delta: float,
    threshold: float,
) -> tuple[list[PersonHit], list[PersonHit]]:
    accepted: dict[str, list[Observation]] = {}
    uncertain: dict[str, Observation] = {}
    for obs in observations:
        if obs.ambiguous:
            previous = uncertain.get(obs.slug)
            if previous is None or obs.score > previous.score:
                uncertain[obs.slug] = obs
            continue
        accepted.setdefault(obs.slug, []).append(obs)

    hits: list[PersonHit] = []
    for slug, items in accepted.items():
        best = max(items, key=lambda item: item.score)
        max_score = best.score
        if kind == "video":
            samples = len({item.frame for item in items})
            strong = max_score >= threshold + strong_delta
            if samples < min_video_hits and not strong:
                continue
        else:
            samples = len(items)
        hits.append(
            PersonHit(
                slug=slug,
                name=items[0].name,
                score=max_score,
                samples=samples,
                clothing_assisted=any(item.clothing_assisted for item in items),
                face_px=best.face_px,
                box=best.box,
            )
        )
    hit_slugs = {hit.slug for hit in hits}
    unsure = [
        PersonHit(obs.slug, obs.name, obs.score, 1, False, obs.face_px, obs.box)
        for obs in uncertain.values()
        if obs.slug not in hit_slugs
    ]
    hits.sort(key=lambda hit: hit.slug)
    unsure.sort(key=lambda hit: hit.slug)
    return hits, unsure
