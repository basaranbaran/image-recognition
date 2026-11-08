"""Head pose from YuNet's five landmarks on an already mirrored frame.

The browser (and only the browser) mirrors the picture so image-left is the
user's left. Yaw is the nose's horizontal offset from the eye midpoint,
divided by the inter-eye distance. Pitch is where the nose sits between the
eyes and the mouth. Bands do not touch: a frontal face cannot fill a turn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from recognize.errors import AppError

# Nose offset / inter-eye distance. Negative: nose toward the user's left.
CENTER_YAW_MAX = 0.10
TURN_YAW_MIN = 0.20
VERTICAL_YAW_MAX = 0.16

# 0 = nose on the eye line, 1 = nose on the mouth line.
# A straight YuNet face sits near 0.58. A slight chin-up reaches about 0.50.
# A slight chin-down reaches about 0.66. The old down floor of 0.70 sat past
# that, so a natural nod was "unknown" until the tilt crushed the landmarks.
# Pitch below 0.36 or above 0.88 is that crushed end, not a usable sample.
CENTER_PITCH_MIN = 0.56
CENTER_PITCH_MAX = 0.62
UP_PITCH_MIN = 0.36
UP_PITCH_MAX = 0.54
DOWN_PITCH_MIN = 0.64
DOWN_PITCH_MAX = 0.88
TURN_PITCH_MIN = 0.28
TURN_PITCH_MAX = 0.72
# Frontal eye-to-mouth span is about one inter-eye distance. Much less means
# the mouth has collapsed into the eyes and the aligned crop is not usable.
MIN_SPAN_OVER_EYES = 0.55

POSE_YAW_LIMIT = 1.2
POSE_PITCH_MIN = -0.15
POSE_PITCH_MAX = 1.35

MIN_FACE_PX = 100
MIN_DET_SCORE = 0.65
CAMERA_DETECT_FLOOR = 0.5
HOLD_FRAMES = 5

_SEEN = {
    "center": "Düz bakıyorsunuz",
    "left": "Sola dönüksünüz",
    "right": "Sağa dönüksünüz",
    "up": "Yukarı bakıyorsunuz",
    "down": "Aşağı bakıyorsunuz",
    "unknown": "Bakış ölçülemedi",
}


@dataclass(frozen=True)
class Step:
    id: str
    instruction: str
    required: bool
    pose: str


STEPS: tuple[Step, ...] = (
    Step("center", "Düz bak", True, "center"),
    Step("left", "Sola dön", True, "left"),
    Step("right", "Sağa dön", True, "right"),
    Step("up", "Biraz yukarı bak", True, "up"),
    Step("down", "Biraz aşağı bak", True, "down"),
    Step("eyes", "Gözlerini kapat", False, "center"),
)


@dataclass(frozen=True)
class ObserveResult:
    step_id: str | None
    instruction: str
    ready: bool
    matching: bool
    streak: int
    hold_frames: int
    required_done: bool
    finished: bool
    optional: bool
    eyes_armed: bool
    message: str


def sequence_line() -> str:
    parts = [
        step.instruction if step.required else f"{step.instruction} (atlanabilir)" for step in STEPS
    ]
    return " · ".join(parts)


def instruction_for(step_id: str) -> str:
    for step in STEPS:
        if step.id == step_id:
            return step.instruction
    raise AppError("Bakış adı geçersiz.")


def yaw_pitch_from_row(row: np.ndarray) -> tuple[float, float]:
    """Yaw and pitch from a YuNet row (14 or 15 floats).

    Landmark order: right eye, left eye, nose, right mouth, left mouth.
    """
    values = np.asarray(row, dtype=np.float64).reshape(-1)
    if values.size < 14:
        raise ValueError("yunet row is short")
    points = [(float(values[4 + 2 * index]), float(values[5 + 2 * index])) for index in range(5)]
    return yaw_pitch(points[0], points[1], points[2], points[3], points[4])


def yaw_pitch(
    right_eye: tuple[float, float],
    left_eye: tuple[float, float],
    nose: tuple[float, float],
    mouth_right: tuple[float, float],
    mouth_left: tuple[float, float],
) -> tuple[float, float]:
    """Measure a mirrored frame. Negative yaw is the user's left."""
    eye_dx = left_eye[0] - right_eye[0]
    eye_dy = left_eye[1] - right_eye[1]
    inter_eye = math.hypot(eye_dx, eye_dy)
    if inter_eye < 1e-3:
        raise ValueError("eyes are coincident")
    eye_mid_x = (right_eye[0] + left_eye[0]) / 2
    eye_mid_y = (right_eye[1] + left_eye[1]) / 2
    mouth_mid_y = (mouth_right[1] + mouth_left[1]) / 2
    span = mouth_mid_y - eye_mid_y
    if span <= 1e-3:
        raise ValueError("mouth is not below the eyes")
    yaw = (nose[0] - eye_mid_x) / inter_eye
    pitch = (nose[1] - eye_mid_y) / span
    return yaw, pitch


def landmarks_hold(
    right_eye: tuple[float, float],
    left_eye: tuple[float, float],
    nose: tuple[float, float],
    mouth_right: tuple[float, float],
    mouth_left: tuple[float, float],
) -> bool:
    """False when the five points have collapsed into a bad capture."""
    del nose
    inter_eye = math.hypot(left_eye[0] - right_eye[0], left_eye[1] - right_eye[1])
    if inter_eye < 1e-3:
        return False
    eye_mid_y = (right_eye[1] + left_eye[1]) / 2
    mouth_mid_y = (mouth_right[1] + mouth_left[1]) / 2
    span = mouth_mid_y - eye_mid_y
    return span >= MIN_SPAN_OVER_EYES * inter_eye


def landmarks_from_row(row: np.ndarray) -> tuple[float, ...]:
    """Five YuNet points as x,y pairs: right eye, left eye, nose, mouth corners."""
    values = np.asarray(row, dtype=np.float64).reshape(-1)
    if values.size < 14:
        raise ValueError("yunet row is short")
    return tuple(float(values[index]) for index in range(4, 14))


def pose_from_yunet(row: np.ndarray) -> tuple[float, float, str]:
    yaw, pitch = yaw_pitch_from_row(row)
    return yaw, pitch, classify_pose(yaw, pitch)


# Baran's saved crops: düz 01.jpg pitch 0.592, yukarı 04.jpg 0.555
# (only -0.037), aşağı 05.jpg 0.654 (+0.062). A 0.06 delta rejects the up.
# Eye position inside those aligned crops moved at most 0.020, so that shift
# is alignment noise. A live nod has to move the eyes further than that.
PITCH_DELTA = 0.03
STRAIGHT_PITCH = 0.025
EYE_DELTA = 0.04
# Re-detected yaw on Baran's mirrored crops: 01.jpg düz +0.031, 02.jpg sol
# -0.329 (delta -0.360), 03.jpg sağ +0.403 (delta +0.372). Positive yaw is the
# user's own right; 03.jpg was stored as kamera-sag.jpg. The aşağı crop
# (05.jpg) drifts only +0.083 and must stay down. A 0.18 turn gate accepts
# the extreme sağ crop, then leaves every smaller turn in the gap where
# 03.jpg's pitch drop (-0.072) is claimed by the up gate.
REL_YAW_STRAIGHT = 0.08
REL_YAW_TURN = 0.09
DOWN_DET_SCORE = 0.50


def classify_from_baseline(
    yaw: float,
    pitch: float,
    neutral_yaw: float,
    neutral_pitch: float,
    eye_frac: float | None = None,
    neutral_eye: float | None = None,
) -> str:
    """Pose from this person's baseline. Yaw, pitch, or the eye line may move."""
    values = [yaw, pitch, neutral_yaw, neutral_pitch]
    if eye_frac is not None and neutral_eye is not None:
        values.extend([eye_frac, neutral_eye])
    if not all(math.isfinite(value) for value in values):
        return "unknown"
    if pitch < UP_PITCH_MIN or pitch > DOWN_PITCH_MAX:
        return "unknown"
    dyaw = yaw - neutral_yaw
    dpitch = pitch - neutral_pitch
    deye = None if eye_frac is None or neutral_eye is None else eye_frac - neutral_eye
    eye_straight = deye is None or abs(deye) < EYE_DELTA
    if abs(dyaw) <= REL_YAW_STRAIGHT and abs(dpitch) <= STRAIGHT_PITCH and eye_straight:
        return "center"
    # Live session (neutral yaw +0.019): a turn to the user's right reached
    # only about +0.03 to +0.08 while pitch fell ~0.07, so the old order
    # called it up. Two frames at delta +0.083 were unknown. Once yaw leaves
    # the straight band it is sağ or sol, even if the chin also drifts.
    # Positive yaw is the user's right (03.jpg kamera-sag.jpg). Do not use a
    # cutoff higher than this band.
    if abs(dyaw) > min(REL_YAW_STRAIGHT, REL_YAW_TURN):
        return "right" if dyaw > 0 else "left"
    chin_up = dpitch <= -PITCH_DELTA or (deye is not None and deye <= -EYE_DELTA)
    chin_down = dpitch >= PITCH_DELTA or (deye is not None and deye >= EYE_DELTA)
    if chin_up and not chin_down:
        return "up"
    if chin_down and not chin_up:
        return "down"
    return "unknown"


def classify_pose(yaw: float, pitch: float) -> str:
    if not math.isfinite(yaw) or not math.isfinite(pitch):
        return "unknown"
    if abs(yaw) > POSE_YAW_LIMIT or pitch < POSE_PITCH_MIN or pitch > POSE_PITCH_MAX:
        return "unknown"
    if abs(yaw) <= VERTICAL_YAW_MAX and UP_PITCH_MIN <= pitch <= UP_PITCH_MAX:
        return "up"
    if abs(yaw) <= VERTICAL_YAW_MAX and DOWN_PITCH_MIN <= pitch <= DOWN_PITCH_MAX:
        return "down"
    if TURN_PITCH_MIN <= pitch <= TURN_PITCH_MAX and yaw <= -TURN_YAW_MIN:
        return "left"
    if TURN_PITCH_MIN <= pitch <= TURN_PITCH_MAX and yaw >= TURN_YAW_MIN:
        return "right"
    if abs(yaw) <= CENTER_YAW_MAX and CENTER_PITCH_MIN <= pitch <= CENTER_PITCH_MAX:
        return "center"
    return "unknown"


class PoseGuide:
    """Accept a step only after the target pose holds for several frames."""

    def __init__(self) -> None:
        self.index = 0
        self.streak = 0
        self.captured_ids: list[str] = []
        self.skipped: set[str] = set()
        self.required_done = False
        self.finished = False
        self.eyes_armed = False

    @property
    def current(self) -> Step | None:
        if self.finished or self.index >= len(STEPS):
            return None
        return STEPS[self.index]

    def observe(self, pose: str, face_px: float, score: float) -> ObserveResult:
        if self.finished:
            return self._result(ready=False, matching=False, message="Bakışlar tamam.")
        step = self.current
        if step is None:
            return self._result(ready=False, matching=False, message="Bakışlar tamam.")
        if step.id == "eyes" and not self.eyes_armed:
            self.streak = 0
            return self._result(
                ready=False,
                matching=False,
                message="Gözlerini kapatın. Hazır olunca düğmeye basın ya da bu adımı atlayın.",
            )
        if face_px < MIN_FACE_PX:
            self.streak = 0
            return self._result(ready=False, matching=False, message="Yüz küçük. Biraz yaklaşın.")
        if score < MIN_DET_SCORE:
            self.streak = 0
            return self._result(ready=False, matching=False, message="Yüz net değil. Işığa dönün.")
        if pose != step.pose:
            self.streak = 0
            seen = _SEEN.get(pose, _SEEN["unknown"])
            return self._result(
                ready=False,
                matching=False,
                message=f"Bu kare kaydedilmedi. {seen}. İstenen: {step.instruction}.",
            )
        if self.streak < HOLD_FRAMES:
            self.streak += 1
        ready = self.streak >= HOLD_FRAMES
        return self._result(
            ready=ready,
            matching=True,
            message=f"Tutun… {self.streak}/{HOLD_FRAMES}",
        )

    def commit(self) -> str:
        step = self.current
        if step is None or self.streak < HOLD_FRAMES:
            raise AppError("Bakış henüz tutulmadı.")
        if step.id == "eyes" and not self.eyes_armed:
            raise AppError("Göz adımı henüz başlamadı.")
        self.captured_ids.append(step.id)
        self.streak = 0
        self.required_done = all(item.id in self.captured_ids for item in STEPS if item.required)
        if self.index + 1 >= len(STEPS):
            self.finished = True
            self.index = len(STEPS)
        else:
            self.index += 1
        return step.id

    def arm_eyes(self) -> None:
        step = self.current
        if step is None or step.id != "eyes":
            raise AppError("Göz adımı henüz gelmedi.")
        self.eyes_armed = True
        self.streak = 0

    def skip(self) -> None:
        step = self.current
        if step is None:
            raise AppError("Atlanacak adım kalmadı.")
        if step.required:
            raise AppError("Bu bakış atlanamaz.")
        self.skipped.add(step.id)
        self.streak = 0
        self.finished = True

    def step_states(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index, step in enumerate(STEPS):
            if step.id in self.captured_ids:
                state = "done"
            elif step.id in self.skipped:
                state = "skipped"
            elif not self.finished and index == self.index:
                state = "current"
            else:
                state = "wait"
            rows.append(
                {
                    "id": step.id,
                    "instruction": step.instruction,
                    "required": step.required,
                    "state": state,
                }
            )
        return rows

    def _result(self, *, ready: bool, matching: bool, message: str) -> ObserveResult:
        step = self.current
        return ObserveResult(
            step_id=None if step is None else step.id,
            instruction="Bakışlar tamam" if step is None else step.instruction,
            ready=ready,
            matching=matching,
            streak=self.streak,
            hold_frames=HOLD_FRAMES,
            required_done=self.required_done,
            finished=self.finished,
            optional=step is not None and not step.required,
            eyes_armed=self.eyes_armed,
            message=message,
        )
