"""Continuous webcam enrollment. Many angles, never one averaged vector.

A normal camera has no depth sensor. This records the face along the path
the head actually travels.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from recognize.pose import classify_from_baseline

_log = logging.getLogger("uvicorn.error")

SWEEP_CAP = 40
MIN_GAP_S = 0.150
YAW_STEP = 0.04
PITCH_STEP = 0.03
STILL_LIMIT = 3
BASELINE_NEED = 4
BASELINE_PITCH_SPREAD = 0.03
BASELINE_YAW_SPREAD = 0.05
FRONTAL_YAW = 0.12
FRONTAL_PITCH_MIN = 0.40
FRONTAL_PITCH_MAX = 0.78

_SEEN = {
    "center": "düz",
    "up": "yukarı",
    "down": "aşağı",
    "left": "sol",
    "right": "sağ",
    "unknown": "belirsiz",
}
_HEAD = {
    "center": "Düz",
    "up": "Yukarı",
    "down": "Aşağı",
    "left": "Sol",
    "right": "Sağ",
    "unknown": "Belirsiz",
}
_MISSING = {
    "left": "sola",
    "right": "sağa",
    "up": "yukarı",
    "down": "aşağı",
    "center": "ortaya",
}

# Coverage quotas. Up and down are measured from this person's neutral, not a fixed ratio.
REGIONS: tuple[tuple[str, str, int], ...] = (
    ("left", "Sola dön", 4),
    ("right", "Sağa dön", 4),
    ("up", "Biraz yukarı bak", 3),
    ("down", "Biraz aşağı bak", 3),
    ("center", "Düz bak", 2),
)
SWEEP_TOTAL = sum(need for _region, _label, need in REGIONS)

# What to do next, said as an action. "Şu an: sol" told people where they were, not where to go.
_NEXT = {
    "left": "Başını yavaşça sola çevir",
    "right": "Başını yavaşça sağa çevir",
    "up": "Çeneni biraz kaldır",
    "down": "Çeneni biraz indir",
    "center": "Kameraya düz bak",
}

# Gaze for the on-screen ring, relative to this person's neutral. A turn that
# classify_from_baseline calls sağ or sol starts at 0.08 yaw; a nod at 0.03 pitch.
# These spans put both near the middle of the ring's reach.
GAZE_YAW_SPAN = 0.20
GAZE_PITCH_SPAN = 0.06


@dataclass(frozen=True)
class SweepRecord:
    yaw: float
    pitch: float
    pose: str
    time: float
    eye: float | None = None


@dataclass(frozen=True)
class SweepTick:
    kept: bool
    covered: bool
    count: int
    message: str


class FaceSweep:
    """Keep distinct views until left, right, up, down, and center are covered."""

    def __init__(self) -> None:
        self.records: list[SweepRecord] = []
        self.neutral_yaw: float | None = None
        self.neutral_pitch: float | None = None
        self.neutral_eye: float | None = None
        self.last_seen = "öğreniliyor"
        self.last_accept = ""
        self._baseline: list[tuple[float, float, float | None, float]] = []

    def lock_neutral(self, yaw: float, pitch: float, eye: float | None = None) -> None:
        self.neutral_yaw = yaw
        self.neutral_pitch = pitch
        self.neutral_eye = eye

    def note(
        self,
        yaw: float,
        pitch: float,
        now: float,
        eye_frac: float | None = None,
    ) -> SweepTick:
        """Store this view when it is not a near-duplicate. Returns whether it was kept."""
        if not self.prepare(yaw, pitch, now, eye_frac):
            return SweepTick(False, self.covered, len(self.records), self.message)
        if not self.accepts(yaw, pitch, now, eye_frac):
            return SweepTick(False, self.covered, len(self.records), self.message)
        return self.keep(yaw, pitch, now, eye_frac)

    def prepare(
        self,
        yaw: float,
        pitch: float,
        now: float,
        eye_frac: float | None = None,
    ) -> bool:
        """True once this person's straight face is known and the frame may be stored."""
        if self.neutral_pitch is None or self.neutral_yaw is None:
            locked = self._feed_baseline(yaw, pitch, now, eye_frac)
            self._touch_seen(yaw, pitch, eye_frac)
            return locked
        self._touch_seen(yaw, pitch, eye_frac)
        return True

    def accepts(
        self,
        yaw: float,
        pitch: float,
        now: float,
        eye_frac: float | None = None,
    ) -> bool:
        if self.neutral_yaw is None or self.neutral_pitch is None:
            return False
        pose = classify_from_baseline(
            yaw,
            pitch,
            self.neutral_yaw,
            self.neutral_pitch,
            eye_frac,
            self.neutral_eye,
        )
        # Finished directions must not keep filling the 40-sample cap.
        # This session stored left 11, right 9, up 13, center 5 and down 2,
        # then refused every later down frame.
        if self._quota_met(pose):
            return False
        return self._accept(yaw, pitch, now, eye_frac)

    def keep(
        self,
        yaw: float,
        pitch: float,
        now: float,
        eye_frac: float | None = None,
    ) -> SweepTick:
        if self.neutral_yaw is None or self.neutral_pitch is None:
            raise RuntimeError("neutral face is not learned")
        pose = classify_from_baseline(
            yaw,
            pitch,
            self.neutral_yaw,
            self.neutral_pitch,
            eye_frac,
            self.neutral_eye,
        )
        if self._quota_met(pose):
            return SweepTick(False, self.covered, len(self.records), self.message)
        self.last_seen = _SEEN[pose]
        if pose == "up":
            self.last_accept = "Yukarı bakış alındı."
        elif pose == "down":
            self.last_accept = "Aşağı bakış alındı."
        elif pose == "right":
            self.last_accept = "Sağa bakış alındı."
        elif pose == "left":
            self.last_accept = "Sola bakış alındı."
        self.records.append(SweepRecord(yaw, pitch, pose, now, eye_frac))
        return SweepTick(True, self.covered, len(self.records), self.message)

    def count(self, region: str) -> int:
        return sum(1 for record in self.records if record.pose == region)

    def _quota_met(self, pose: str) -> bool:
        """Unknown poses are not a coverage region, so they are not stored."""
        for region, _label, need in REGIONS:
            if region == pose:
                return self.count(region) >= need
        return True

    @property
    def covered(self) -> bool:
        return all(self.count(region) >= need for region, _label, need in REGIONS)

    def coverage(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for region, label, need in REGIONS:
            count = self.count(region)
            rows.append(
                {
                    "id": region,
                    "label": label,
                    "count": count,
                    "need": need,
                    "done": count >= need,
                }
            )
        return rows

    @property
    def headline(self) -> str:
        if self.neutral_pitch is None:
            return "Önce düz bak, yüzünü öğreniyorum"
        if self.covered:
            return "Yüz yapısı tamam"
        region = self.next_region
        return _NEXT[region] if region else f"Şu an: {_HEAD.get(self._pose_key(), 'Belirsiz')}"

    @property
    def next_region(self) -> str | None:
        """Keep going where the head already is; otherwise the first unfinished direction."""
        missing = [region for region, _label, need in REGIONS if self.count(region) < need]
        if not missing:
            return None
        current = self._pose_key()
        if current in missing:
            return current
        return missing[0]

    def pose_of(self, yaw: float, pitch: float, eye_frac: float | None = None) -> str:
        """Direction relative to this person's neutral; unknown until it is learned."""
        if self.neutral_yaw is None or self.neutral_pitch is None:
            return "unknown"
        return classify_from_baseline(
            yaw,
            pitch,
            self.neutral_yaw,
            self.neutral_pitch,
            eye_frac,
            self.neutral_eye,
        )

    def gaze(
        self,
        yaw: float,
        pitch: float,
        eye_frac: float | None = None,
    ) -> tuple[float, float] | None:
        """Head direction on a unit disc: x toward the user's right, y up. None before neutral."""
        if self.neutral_yaw is None or self.neutral_pitch is None:
            return None
        x = (yaw - self.neutral_yaw) / GAZE_YAW_SPAN
        lift = -(pitch - self.neutral_pitch) / GAZE_PITCH_SPAN
        if eye_frac is not None and self.neutral_eye is not None:
            eye_lift = -(eye_frac - self.neutral_eye) / (2 * GAZE_PITCH_SPAN)
            lift = lift if abs(lift) >= abs(eye_lift) else eye_lift
        length = math.hypot(x, lift)
        if length > 1.0:
            x, lift = x / length, lift / length
        return x, lift

    @property
    def message(self) -> str:
        if self.neutral_pitch is None:
            return "Önce düz bak, yüzünü öğreniyorum."
        if self.covered:
            return "Yüz yapısı tamam. Dönüş boyunca alınan açılar ayrı ayrı saklanır."
        missing = [
            _MISSING[region] for region, _label, need in REGIONS if self.count(region) < need
        ]
        if not missing:
            return "Yüz yapısı tamam. Dönüş boyunca alınan açılar ayrı ayrı saklanır."
        prefix = f"{self.last_accept} " if self.last_accept else ""
        return f"{prefix}Şu an: {self.last_seen}. Eksik: {', '.join(missing)}."

    def _pose_key(self) -> str:
        for key, word in _SEEN.items():
            if word == self.last_seen:
                return key
        return "unknown"

    def _touch_seen(self, yaw: float, pitch: float, eye_frac: float | None = None) -> None:
        if self.neutral_yaw is None or self.neutral_pitch is None:
            self.last_seen = "düz" if self._looks_frontal(yaw, pitch) else "belirsiz"
            return
        pose = classify_from_baseline(
            yaw,
            pitch,
            self.neutral_yaw,
            self.neutral_pitch,
            eye_frac,
            self.neutral_eye,
        )
        self.last_seen = _SEEN[pose]
        if pose != "center":
            _log.debug(
                "kare yaw=%+.3f pitch=%.3f pose=%s neutral_yaw=%+.3f",
                yaw,
                pitch,
                pose,
                self.neutral_yaw,
            )

    def _looks_frontal(self, yaw: float, pitch: float) -> bool:
        return abs(yaw) <= FRONTAL_YAW and FRONTAL_PITCH_MIN <= pitch <= FRONTAL_PITCH_MAX

    def _feed_baseline(
        self,
        yaw: float,
        pitch: float,
        now: float,
        eye_frac: float | None = None,
    ) -> bool:
        if not self._looks_frontal(yaw, pitch):
            self._baseline.clear()
            return False
        self._baseline.append((yaw, pitch, eye_frac, now))
        recent = self._baseline[-BASELINE_NEED:]
        if len(recent) < BASELINE_NEED:
            return False
        yaws = [item[0] for item in recent]
        pitches = [item[1] for item in recent]
        if max(pitches) - min(pitches) > BASELINE_PITCH_SPREAD:
            return False
        if max(yaws) - min(yaws) > BASELINE_YAW_SPREAD:
            return False
        self.neutral_yaw = sum(yaws) / len(yaws)
        self.neutral_pitch = sum(pitches) / len(pitches)
        eyes = [item[2] for item in recent if item[2] is not None]
        if len(eyes) == len(recent):
            self.neutral_eye = sum(eyes) / len(eyes)
        return True

    def _accept(
        self,
        yaw: float,
        pitch: float,
        now: float,
        eye_frac: float | None = None,
    ) -> bool:
        if len(self.records) >= SWEEP_CAP:
            return False
        if not self.records:
            return True
        last = self.records[-1]
        tiny = abs(yaw - last.yaw) < YAW_STEP and abs(pitch - last.pitch) < PITCH_STEP
        if eye_frac is not None and last.eye is not None and abs(eye_frac - last.eye) >= 0.02:
            tiny = False
        if tiny and now - last.time < MIN_GAP_S:
            return False
        if tiny and self._near(yaw, pitch) >= STILL_LIMIT:
            return False
        return True

    def _near(self, yaw: float, pitch: float) -> int:
        return sum(
            1
            for record in self.records
            if abs(record.yaw - yaw) < YAW_STEP and abs(record.pitch - pitch) < PITCH_STEP
        )
