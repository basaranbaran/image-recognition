"""Progress units and the time estimate the panel shows during a scan.

Files are scanned in path order, so a pool does all photos first and the videos last.
A per-file average would then promise an end that the videos push away. Each photo
counts as one unit; a video counts as its number of sampled frames times a weight.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from recognize.media import VIDEO_EXTENSIONS

ETA_MIN_FILES = 3
ETA_MIN_SECONDS = 5.0
ETA_ROUND_SECONDS = 5
VIDEO_FRAME_WEIGHT = 0.5


def video_duration(path: Path) -> float | None:
    """Seconds, from the container header only. None when the header does not say."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if fps <= 0 or count <= 0:
        return None
    return count / fps


def frames_sampled(duration: float | None, sample_seconds: float) -> int:
    if duration is None or duration <= 0:
        return 1
    return max(1, math.ceil(duration / max(sample_seconds, 0.2)))


@dataclass
class Plan:
    """Units per file. The video weight is recalibrated once the first video finishes."""

    frames: dict[Path, int] = field(default_factory=dict)
    photos: int = 0
    weight: float = VIDEO_FRAME_WEIGHT
    photo_seconds: float | None = None
    photos_done: int = 0
    frames_done: int = 0
    files_done: int = 0
    _current: Path | None = None
    _current_frames: int = 0

    @property
    def total_units(self) -> float:
        return self.photos + sum(self.frames.values()) * self.weight

    @property
    def done_units(self) -> float:
        return self.photos_done + self.frames_done * self.weight

    def start_file(self, path: Path) -> None:
        self._current = path
        self._current_frames = 0

    def frame(self) -> None:
        """One sampled video frame finished: advance by its weight, never past the file."""
        path = self._current
        if path is None or path not in self.frames:
            return
        if self._current_frames < self.frames[path]:
            self._current_frames += 1
            self.frames_done += 1

    def finish_file(self, path: Path, seconds: float) -> None:
        """Account for the whole file and, for a video, learn how heavy a frame really is."""
        self.files_done += 1
        planned = self.frames.get(path)
        if planned is not None:
            self.frames_done += max(0, planned - self._current_frames)
            if self.photo_seconds and planned > 0 and seconds > 0:
                observed = seconds / planned / self.photo_seconds
                self.weight = min(4.0, max(0.1, observed))
        else:
            self.photos_done += 1
            if seconds > 0:
                self.photo_seconds = (
                    seconds
                    if self.photo_seconds is None
                    else self.photo_seconds * 0.8 + seconds * 0.2
                )
        self._current = None
        self._current_frames = 0


def plan_units(
    files: list[Path],
    sample_seconds: float,
    *,
    duration_of: Callable[[Path], float | None] = video_duration,
) -> Plan:
    plan = Plan()
    for path in files:
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            plan.frames[path] = frames_sampled(duration_of(path), sample_seconds)
        else:
            plan.photos += 1
    return plan


def estimate_eta(
    done_units: float,
    total_units: float,
    elapsed_seconds: float,
    files_done: int,
) -> int | None:
    """Seconds left, rounded up to 5, or None while there is too little to go on."""
    if files_done < ETA_MIN_FILES and elapsed_seconds < ETA_MIN_SECONDS:
        return None
    if done_units <= 0 or total_units <= 0 or elapsed_seconds <= 0:
        return None
    remaining = max(0.0, total_units - done_units)
    rate = done_units / elapsed_seconds
    if rate <= 0:
        return None
    seconds = remaining / rate
    return int(math.ceil(seconds / ETA_ROUND_SECONDS) * ETA_ROUND_SECONDS)
