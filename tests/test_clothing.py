"""Clothing hue masks. No face model required."""

import cv2
import numpy as np

from recognize.clothing import (
    clothing_matches,
    fraction_in_hint,
    hint_for,
    torso_bounds,
)


def _bgr(hue: int, saturation: int, value: int, width: int = 20, height: int = 20) -> np.ndarray:
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[:, :] = (hue, saturation, value)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_yellow_patch_is_in_range_and_blue_is_not() -> None:
    yellow = hint_for("sarı")
    assert yellow is not None
    assert yellow.hue_min == 10
    assert yellow.hue_max == 25
    assert fraction_in_hint(_bgr(18, 200, 220), yellow) > 0.9
    assert fraction_in_hint(_bgr(120, 200, 220), yellow) == 0.0


def test_yellow_fraction_threshold() -> None:
    hint = hint_for("sari")
    assert hint is not None
    yellow = _bgr(18, 220, 220, 1, 1)[0, 0]
    blue = _bgr(120, 220, 220, 1, 1)[0, 0]
    mostly = np.empty((10, 10, 3), dtype=np.uint8)
    mostly[:, :] = blue
    mostly[:4, :] = yellow
    assert fraction_in_hint(mostly, hint) >= 0.35
    sparse = np.empty((10, 10, 3), dtype=np.uint8)
    sparse[:, :] = blue
    sparse[:3, :] = yellow
    assert fraction_in_hint(sparse, hint) < 0.35


def test_yellow_torso_under_the_face_matches_and_blue_does_not() -> None:
    hint = hint_for("sari")
    assert hint is not None
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    face = (80.0, 20.0, 40.0, 40.0)
    x1, y1, x2, y2 = torso_bounds(*face, 200, 200)
    assert (x1, y1, x2, y2) == (76, 60, 124, 140)
    frame[y1:y2, x1:x2] = _bgr(18, 220, 220, x2 - x1, y2 - y1)
    assert clothing_matches(frame, face, hint)
    frame[y1:y2, x1:x2] = _bgr(120, 220, 220, x2 - x1, y2 - y1)
    assert not clothing_matches(frame, face, hint)


def test_torso_below_the_frame_does_not_match() -> None:
    hint = hint_for("sari")
    assert hint is not None
    frame = _bgr(18, 220, 220, 80, 80)
    assert not clothing_matches(frame, (10.0, 60.0, 20.0, 30.0), hint)


def test_named_colors_round_trip_aliases() -> None:
    assert hint_for("SARI") == hint_for("sarı")
    red = hint_for("kırmızı")
    assert red is not None
    assert fraction_in_hint(_bgr(176, 200, 220), red) > 0.9
    assert hint_for("yok") is None
