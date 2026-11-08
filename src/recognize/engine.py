"""YuNet detection and SFace embeddings. Side views are not filtered out."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from recognize.clothing import ClothingHint, clothing_matches
from recognize.errors import AppError
from recognize.gallery import Person
from recognize.match import (
    ANCHOR_MIN_PX,
    ANCHOR_SCORE,
    SMALL_FACE_PX,
    FaceDecision,
    Observation,
    PersonHit,
    ScanOptions,
    ScoredPerson,
    decide_ranked,
    l2_normalize,
    needs_clothing_check,
    rank_people,
    summarize,
)
from recognize.media import iter_video_frames, load_image, resize_max_edge
from recognize.models_io import ensure_models


@dataclass
class DetectedFace:
    row: np.ndarray
    score: float
    x: float
    y: float
    width: float
    height: float

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.width, self.height)


@dataclass
class Inspection:
    hits: list[PersonHit]
    uncertain: list[PersonHit]


class UnreadableMedia(AppError):
    """The container opened but not one frame decoded: a half-copied or header-only video."""


class FaceEngine:
    def __init__(self, model_directory: Path, log=None) -> None:
        yunet, sface = ensure_models(model_directory, log)
        self._detector = cv2.FaceDetectorYN.create(
            str(yunet),
            "",
            (320, 320),
            0.5,
            0.3,
            5000,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(sface), "")

    def detect(self, image: np.ndarray, score_threshold: float) -> list[DetectedFace]:
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        self._detector.setScoreThreshold(float(score_threshold))
        raw = self._detector.detect(image)
        matrix = _unwrap_faces(raw)
        if matrix is None or len(matrix) == 0:
            return []
        faces: list[DetectedFace] = []
        for row in matrix:
            values = np.asarray(row, dtype=np.float32).reshape(-1)
            x = float(values[0])
            y = float(values[1])
            box_w = float(values[2])
            box_h = float(values[3])
            aligned = values[:14] if values.size >= 14 else values
            faces.append(
                DetectedFace(
                    row=aligned,
                    score=float(values[-1]),
                    x=x,
                    y=y,
                    width=box_w,
                    height=box_h,
                )
            )
        return faces

    def embed(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        aligned = self.align(image, face)
        feature = np.asarray(self._recognizer.feature(aligned), dtype=np.float32).reshape(-1)
        return l2_normalize(feature)

    def align(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        return self._recognizer.alignCrop(image, face.row)

    def crop_jpeg(self, image: np.ndarray, face: DetectedFace, quality: int = 85) -> bytes:
        aligned = self.align(image, face)
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()


class GalleryRecognizer:
    """Match one file against the enrolled gallery."""

    def __init__(
        self,
        engine: FaceEngine,
        people: list[Person],
        options: ScanOptions,
        targets: set[str] | None = None,
    ) -> None:
        self.engine = engine
        self.people = people
        self.options = options
        # Faces are ranked against everyone; a video may stop once these slugs are confirmed.
        self.targets = targets
        # Every photo face of this scan, kept for the second look in event_matches.
        self._seen: list[tuple[Path, float, np.ndarray]] = []

    def inspect_image(self, path: Path, progress=None) -> Inspection:
        del progress
        image = resize_max_edge(load_image(path), self.options.image_max_edge)
        observations = self._observe(image, frame_index=0, remember=path)
        hits, uncertain = summarize(
            observations,
            kind="image",
            min_video_hits=self.options.min_video_hits,
            strong_delta=self.options.strong_delta,
            threshold=self.options.threshold,
        )
        return Inspection(hits, uncertain)

    def inspect_video(self, path: Path, progress=None) -> Inspection:
        observations: list[Observation] = []
        enrolled = self.targets or {person.slug for person in self.people}
        frames = 0
        for index, frame in enumerate(iter_video_frames(path, self.options.sample_seconds)):
            frames += 1
            if progress is not None:
                progress(f"{path.name} · {index + 1}. kare")
            resized = resize_max_edge(frame, self.options.video_max_edge)
            observations.extend(self._observe(resized, frame_index=index))
            hits, _uncertain = summarize(
                observations,
                kind="video",
                min_video_hits=self.options.min_video_hits,
                strong_delta=self.options.strong_delta,
                threshold=self.options.threshold,
            )
            if enrolled and {hit.slug for hit in hits} >= enrolled:
                break
        if frames == 0:
            raise UnreadableMedia(f"Videodan kare okunamadı: {path.name}")
        hits, uncertain = summarize(
            observations,
            kind="video",
            min_video_hits=self.options.min_video_hits,
            strong_delta=self.options.strong_delta,
            threshold=self.options.threshold,
        )
        return Inspection(hits, uncertain)

    def event_matches(self) -> dict[Path, list[PersonHit]]:
        """Look at the photos again with each person's clearest event faces as references.

        An anchor never vouches for its own photo, and learned scores only count for
        faces of at least SMALL_FACE_PX. Nothing is written to the saved gallery.
        """
        options = self.options
        if not options.learn_from_event or not self._seen or not self.people:
            return {}
        paths = [item[0] for item in self._seen]
        path_ids = {path: number for number, path in enumerate(dict.fromkeys(paths))}
        ids = np.array([path_ids[path] for path in paths])
        px = np.array([item[1] for item in self._seen], dtype=np.float32)
        faces = np.stack([item[2] for item in self._seen])
        base = np.stack(
            [(faces @ person.embeddings.T).max(axis=1) for person in self.people], axis=1
        )
        owner = base.argmax(axis=1)
        learned = np.full_like(base, -1.0)
        for column in range(len(self.people)):
            clear = np.flatnonzero(
                (owner == column) & (base[:, column] >= ANCHOR_SCORE) & (px >= ANCHOR_MIN_PX)
            )
            best: dict[int, int] = {}
            for index in clear:
                current = best.get(int(ids[index]))
                if current is None or base[index, column] > base[current, column]:
                    best[int(ids[index])] = int(index)
            anchors = np.array(list(best.values()), dtype=int)
            if anchors.size == 0:
                continue
            similarity = faces @ faces[anchors].T
            similarity[ids[:, None] == ids[anchors][None, :]] = -1.0
            learned[:, column] = similarity.max(axis=1)
        learned[px < SMALL_FACE_PX] = -1.0
        scores = np.maximum(base, learned)
        found: dict[Path, dict[str, PersonHit]] = {}
        for index, path in enumerate(paths):
            ranked = sorted(
                (
                    ScoredPerson(person.slug, person.name, float(scores[index, column]), False)
                    for column, person in enumerate(self.people)
                ),
                key=lambda item: item.score,
                reverse=True,
            )
            decision = decide_ranked(
                ranked,
                threshold=options.threshold,
                margin=options.margin,
                weak_band=options.weak_band,
                clothing_matches=False,
            )
            if decision is None or decision.ambiguous:
                continue
            hits = found.setdefault(path, {})
            previous = hits.get(decision.slug)
            if previous is None or decision.score > previous.score:
                hits[decision.slug] = PersonHit(
                    decision.slug, decision.name, decision.score, 1, False, float(px[index])
                )
        return {path: list(hits.values()) for path, hits in found.items()}

    def _observe(
        self,
        image: np.ndarray,
        frame_index: int,
        remember: Path | None = None,
    ) -> list[Observation]:
        options = self.options
        gallery = [
            (person.slug, person.name, person.embeddings, person.clothing is not None)
            for person in self.people
        ]
        found: list[Observation] = []
        height, width = image.shape[:2]
        for face in self.engine.detect(image, options.min_det_score):
            if min(face.width, face.height) < options.min_face_px:
                continue
            box = _normalised_box(face, width, height)
            try:
                embedding = self.engine.embed(image, face)
            except (cv2.error, ValueError):
                continue
            if remember is not None:
                self._seen.append((remember, float(min(face.width, face.height)), embedding))
            ranked = rank_people(embedding, gallery)
            check = needs_clothing_check(
                ranked,
                threshold=options.threshold,
                margin=options.margin,
                weak_band=options.weak_band,
            )
            matched = False
            if check and ranked:
                person = _person_by_slug(self.people, ranked[0].slug)
                hint: ClothingHint | None = None if person is None else person.clothing
                if hint is not None:
                    matched = clothing_matches(image, face.xywh, hint)
            decision: FaceDecision | None = decide_ranked(
                ranked,
                threshold=options.threshold,
                margin=options.margin,
                weak_band=options.weak_band,
                clothing_matches=matched,
            )
            if decision is None:
                continue
            found.append(
                Observation(
                    frame=frame_index,
                    slug=decision.slug,
                    name=decision.name,
                    score=decision.score,
                    clothing_assisted=decision.clothing_assisted,
                    ambiguous=decision.ambiguous,
                    face_px=float(min(face.width, face.height)),
                    box=box,
                )
            )
        return found


def _normalised_box(
    face: DetectedFace, width: int, height: int
) -> tuple[float, float, float, float]:
    """The box as fractions of the frame, so it survives any later resize of the picture."""
    if width <= 0 or height <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    x = min(max(face.x / width, 0.0), 1.0)
    y = min(max(face.y / height, 0.0), 1.0)
    w = min(max(face.width / width, 0.0), 1.0 - x)
    h = min(max(face.height / height, 0.0), 1.0 - y)
    return (round(x, 4), round(y, 4), round(w, 4), round(h, 4))


def _person_by_slug(people: list[Person], slug: str) -> Person | None:
    for person in people:
        if person.slug == slug:
            return person
    return None


def _unwrap_faces(raw: object) -> np.ndarray | None:
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        return raw
    if isinstance(raw, tuple):
        for item in raw:
            if isinstance(item, np.ndarray):
                return item
    return None
