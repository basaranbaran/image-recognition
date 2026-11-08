"""Camera enrollment. The image is already mirrored; nothing is sent off this machine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from recognize.engine import DetectedFace
from recognize.enroll import enroll_faces
from recognize.errors import AppError
from recognize.gallery import Person, ReferenceSample, find_person
from recognize.pose import (
    CAMERA_DETECT_FLOOR,
    DOWN_DET_SCORE,
    MIN_DET_SCORE,
    MIN_FACE_PX,
    PoseGuide,
    instruction_for,
    landmarks_from_row,
    landmarks_hold,
    pose_from_yunet,
)
from recognize.sweep import FaceSweep

BLUR_VARIANCE_MIN = 40.0
# A second detection only counts as another person when it is confident and at
# least half the size of the main face. At the 0.50 detection floor YuNet also
# fires on posters, reflections and wall patterns behind a lone user, which used
# to stop the sweep with "Kadrajda tek yüz olsun".
SECOND_FACE_SCORE = 0.75
SECOND_FACE_SHARE = 0.5

# Mean gray inside the face box. Webcams lift gain in a dim room, so a face
# below this is mostly sensor noise and its landmarks wander.
DARK_MEAN_MAX = 50.0

_SOURCE_NAMES = {
    "center": "kamera-duz.jpg",
    "left": "kamera-sol.jpg",
    "right": "kamera-sag.jpg",
    "up": "kamera-yukari.jpg",
    "down": "kamera-asagi.jpg",
    "eyes": "kamera-goz.jpg",
}


class EnrollmentEngine(Protocol):
    def detect(self, image: np.ndarray, score_threshold: float) -> list[DetectedFace]: ...

    def embed(self, image: np.ndarray, face: DetectedFace) -> np.ndarray: ...

    def crop_jpeg(self, image: np.ndarray, face: DetectedFace, quality: int = 85) -> bytes: ...


@dataclass
class CameraSession:
    name: str
    samples: list[tuple[np.ndarray, float, bytes, str] | ReferenceSample]
    guide: PoseGuide | None = None
    sweep: FaceSweep | None = None
    saved: int = 0


@dataclass(frozen=True)
class FrameReport:
    pose: str
    yaw: float | None
    pitch: float | None
    face_px: float
    score: float
    captured: bool
    captured_label: str | None
    crop: bytes | None
    matches: bool
    message: str
    ready: bool = False
    coverage: tuple[dict[str, object], ...] = ()
    # Machine-readable reason, so the page can wait before showing a passing warning.
    hint: str = ""
    # Face box as fractions of the mirrored frame: x, y, width, height.
    box: tuple[float, float, float, float] | None = None
    # Head direction relative to neutral on a unit disc; x is the user's right, y is up.
    gaze: tuple[float, float] | None = None


def source_name(step_id: str) -> str:
    try:
        return _SOURCE_NAMES[step_id]
    except KeyError as exc:
        raise AppError("Bakış adı geçersiz.") from exc


def assess_frame(
    engine: EnrollmentEngine,
    session: CameraSession,
    image: np.ndarray,
) -> FrameReport:
    """Score one mirrored frame. Callers serialize access to ``session``."""
    if session.guide.finished:
        return _quiet("Bakışlar tamam.")
    face, hint = _main_face(list(engine.detect(image, CAMERA_DETECT_FLOOR)))
    if face is None:
        session.guide.observe("unknown", 0, 0)
        if hint == "no_face":
            message = "Yüz görünmüyor. Kameraya bakın."
        else:
            message = "Kadrajda tek yüz olsun."
        return _quiet(message)
    try:
        yaw, pitch, pose = pose_from_yunet(face.row)
    except ValueError:
        yaw, pitch, pose = None, None, "unknown"
    face_px = float(min(face.width, face.height))
    observed = session.guide.observe(pose, face_px, face.score)
    if not observed.ready:
        return FrameReport(
            pose=pose,
            yaw=yaw,
            pitch=pitch,
            face_px=face_px,
            score=float(face.score),
            captured=False,
            captured_label=None,
            crop=None,
            matches=observed.matching,
            message=observed.message,
        )
    try:
        embedding = np.asarray(engine.embed(image, face), dtype=np.float32).reshape(-1)
        jpeg = engine.crop_jpeg(image, face)
    except (cv2.error, ValueError):
        return FrameReport(
            pose=pose,
            yaw=yaw,
            pitch=pitch,
            face_px=face_px,
            score=float(face.score),
            captured=False,
            captured_label=None,
            crop=None,
            matches=True,
            message="Bu kareden yüz vektörü çıkmadı. Tutmaya devam edin.",
        )
    captured_id = session.guide.commit()
    label = instruction_for(captured_id)
    session.samples.append((embedding, float(face.score), jpeg, source_name(captured_id)))
    return FrameReport(
        pose=pose,
        yaw=yaw,
        pitch=pitch,
        face_px=face_px,
        score=float(face.score),
        captured=True,
        captured_label=label,
        crop=jpeg,
        matches=False,
        message=f"{label} alındı.",
    )


def record_sweep_frame(
    engine: EnrollmentEngine,
    session: CameraSession,
    image: np.ndarray,
    now: float,
) -> FrameReport:
    """Keep one view of a moving face when it adds a new angle."""
    sweep = session.sweep
    if sweep is None:
        raise AppError("Tarama oturumu yok.")
    if sweep.covered:
        return _sweep_report(
            sweep, pose="unknown", yaw=None, pitch=None, matches=False, hint="done"
        )
    face, hint = _main_face(list(engine.detect(image, DOWN_DET_SCORE)))
    if face is None:
        if hint == "no_face":
            text = "Yüz görünmüyor. Kameraya bakın."
        else:
            text = "Yanınızda başka bir yüz var. Kadrajda tek yüz olsun."
        return _sweep_report(
            sweep,
            pose="unknown",
            yaw=None,
            pitch=None,
            matches=False,
            message=text,
            hint=hint,
        )
    box = _box_fraction(image, face)
    face_px = float(min(face.width, face.height))

    def refuse(
        message: str,
        hint: str,
        *,
        pose: str = "unknown",
        yaw: float | None = None,
        pitch: float | None = None,
        gaze: tuple[float, float] | None = None,
    ) -> FrameReport:
        return _sweep_report(
            sweep,
            pose=pose,
            yaw=yaw,
            pitch=pitch,
            matches=False,
            message=message,
            hint=hint,
            box=box,
            gaze=gaze,
        )

    if face_px < MIN_FACE_PX:
        return refuse("Yüz küçük. Biraz yaklaşın.", "small")
    gray = _face_gray(image, face)
    if gray is None:
        return refuse("Yüz kadrajın kenarında. Ortaya gelin.", "edge")
    if float(gray.mean()) < DARK_MEAN_MAX:
        return refuse("Yüz karanlıkta. Işığa dönün.", "dark")
    if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < BLUR_VARIANCE_MIN:
        return refuse("Görüntü bulanık. Biraz daha yavaş dönün.", "blur")
    try:
        yaw, pitch, _absolute = pose_from_yunet(face.row)
        landmarks = landmarks_from_row(face.row)
        points = tuple((landmarks[index], landmarks[index + 1]) for index in range(0, 10, 2))
        held = landmarks_hold(points[0], points[1], points[2], points[3], points[4])
    except ValueError:
        return refuse("Bakış ölçülemedi. Yüzünüzü kadrajda tutun.", "landmarks")
    eye_frac = ((landmarks[1] + landmarks[3]) / 2.0) / float(image.shape[0])
    gaze = sweep.gaze(yaw, pitch, eye_frac)
    pose = sweep.pose_of(yaw, pitch, eye_frac)
    if not held:
        return refuse(
            "Yüz noktaları dağıldı. Biraz daha az eğin.",
            "landmarks",
            pose=pose,
            yaw=yaw,
            pitch=pitch,
            gaze=gaze,
        )
    score_floor = MIN_DET_SCORE
    if _chin_down(sweep, pitch, eye_frac):
        score_floor = DOWN_DET_SCORE
    if float(face.score) < score_floor:
        return refuse(
            "Yüz net değil. Işığa dönün.",
            "weak",
            pose=pose,
            yaw=yaw,
            pitch=pitch,
            gaze=gaze,
        )
    if not sweep.prepare(yaw, pitch, now, eye_frac):
        return _sweep_report(
            sweep,
            pose=pose,
            yaw=yaw,
            pitch=pitch,
            matches=True,
            hint="learning",
            box=box,
        )
    pose = sweep.pose_of(yaw, pitch, eye_frac)
    gaze = sweep.gaze(yaw, pitch, eye_frac)
    if not sweep.accepts(yaw, pitch, now, eye_frac):
        return _sweep_report(
            sweep,
            pose=pose,
            yaw=yaw,
            pitch=pitch,
            matches=True,
            hint="tracking",
            box=box,
            gaze=gaze,
        )
    try:
        embedding = np.asarray(engine.embed(image, face), dtype=np.float32).reshape(-1)
        jpeg = engine.crop_jpeg(image, face)
    except (cv2.error, ValueError):
        return _sweep_report(
            sweep,
            pose=pose,
            yaw=yaw,
            pitch=pitch,
            matches=True,
            message="Bu kareden yüz vektörü çıkmadı. Dönmeye devam edin.",
            hint="tracking",
            box=box,
            gaze=gaze,
        )
    tick = sweep.keep(yaw, pitch, now, eye_frac)
    kept_pose = sweep.records[-1].pose
    label = _sweep_label(kept_pose)
    session.samples.append(
        ReferenceSample(
            embedding,
            float(face.score),
            jpeg,
            f"kamera-{kept_pose}-{tick.count:02d}.jpg",
            yaw,
            pitch,
            landmarks,
        )
    )
    return _sweep_report(
        sweep,
        pose=kept_pose,
        yaw=yaw,
        pitch=pitch,
        matches=True,
        captured=True,
        captured_label=label,
        crop=jpeg,
        face_px=face_px,
        score=float(face.score),
        hint="captured",
        box=box,
        gaze=gaze,
    )


def save_enrollment(session: CameraSession, root: Path | None = None) -> tuple[Person, str | None]:
    """Append held poses. No clothing hint is created or replaced."""
    if session.sweep is not None:
        if not session.sweep.covered:
            raise AppError("Yüz yapısı tamamlanmadan kayıt yazılmaz.")
    elif session.guide is None or not session.guide.required_done:
        raise AppError("Zorunlu bakışlar bitmeden kayıt yazılmaz.")
    start = session.saved
    pending = list(session.samples[start:])
    if not pending:
        person = find_person(session.name, root)
        if person is None:
            raise AppError("Kaydedilecek yeni bakış yok.")
        return person, None
    person, warning = enroll_faces(
        session.name,
        pending,
        None,
        update_clothing=False,
        root=root,
    )
    session.saved = start + len(pending)
    return person, warning


def _chin_down(sweep: FaceSweep, pitch: float, eye_frac: float) -> bool:
    if sweep.neutral_pitch is None:
        return False
    if pitch - sweep.neutral_pitch >= 0.02:
        return True
    if sweep.neutral_eye is not None and eye_frac - sweep.neutral_eye >= 0.02:
        return True
    return False


def _main_face(faces: list[DetectedFace]) -> tuple[DetectedFace | None, str]:
    """The largest face, unless a real second person shares the frame."""
    if not faces:
        return None, "no_face"
    ordered = sorted(faces, key=lambda item: item.width * item.height, reverse=True)
    main = ordered[0]
    size = min(main.width, main.height)
    for other in ordered[1:]:
        if other.score >= SECOND_FACE_SCORE and min(other.width, other.height) >= (
            SECOND_FACE_SHARE * size
        ):
            return None, "many_faces"
    return main, ""


def _face_gray(image: np.ndarray, face: DetectedFace) -> np.ndarray | None:
    height, width = image.shape[:2]
    x0 = max(0, int(face.x))
    y0 = max(0, int(face.y))
    x1 = min(width, int(face.x + face.width))
    y1 = min(height, int(face.y + face.height))
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    return cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)


def _box_fraction(image: np.ndarray, face: DetectedFace) -> tuple[float, float, float, float]:
    height, width = image.shape[:2]
    return (
        round(face.x / width, 4),
        round(face.y / height, 4),
        round(face.width / width, 4),
        round(face.height / height, 4),
    )


def _sweep_label(pose: str) -> str:
    """Confirmation for a stored view. The instruction itself is the sweep headline."""
    labels = {
        "left": "Sola bakış alındı.",
        "right": "Sağa bakış alındı.",
        "up": "Yukarı bakış alındı.",
        "down": "Aşağı bakış alındı.",
        "center": "Düz bakış alındı.",
    }
    return labels.get(pose, "Ara açı alındı.")


def _sweep_report(
    sweep: FaceSweep,
    *,
    pose: str,
    yaw: float | None,
    pitch: float | None,
    matches: bool,
    message: str | None = None,
    captured: bool = False,
    captured_label: str | None = None,
    crop: bytes | None = None,
    face_px: float = 0,
    score: float = 0,
    hint: str = "",
    box: tuple[float, float, float, float] | None = None,
    gaze: tuple[float, float] | None = None,
) -> FrameReport:
    return FrameReport(
        pose=pose,
        yaw=yaw,
        pitch=pitch,
        face_px=face_px,
        score=score,
        captured=captured,
        captured_label=captured_label,
        crop=crop,
        matches=matches,
        message=sweep.message if message is None else message,
        ready=sweep.covered,
        coverage=tuple(sweep.coverage()),
        hint=hint,
        box=box,
        gaze=gaze,
    )


def _quiet(message: str) -> FrameReport:
    return FrameReport(
        pose="unknown",
        yaw=None,
        pitch=None,
        face_px=0,
        score=0,
        captured=False,
        captured_label=None,
        crop=None,
        matches=False,
        message=message,
    )
