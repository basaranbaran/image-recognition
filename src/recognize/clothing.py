"""Optional clothing hint. It never identifies a person by itself."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from recognize.errors import AppError
from recognize.textutil import tr_lower

MIN_CLOTHING_FRACTION = 0.35

_COLOR_ALIASES = {
    "sari": "sari",
    "sarı": "sari",
    "yellow": "sari",
    "kirmizi": "kirmizi",
    "kırmızı": "kirmizi",
    "red": "kirmizi",
    "mavi": "mavi",
    "blue": "mavi",
    "yesil": "yesil",
    "yeşil": "yesil",
    "green": "yesil",
    "siyah": "siyah",
    "black": "siyah",
    "beyaz": "beyaz",
    "white": "beyaz",
    "yok": "",
    "none": "",
    "yoktur": "",
}


@dataclass(frozen=True)
class ClothingHint:
    """Hue range in OpenCV HSV (H 0–179). Matching reads these numbers, not the label."""

    name: str
    label: str
    hue_min: int
    hue_max: int
    hue_min2: int | None
    hue_max2: int | None
    s_min: int
    s_max: int | None
    v_min: int
    v_max: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label": self.label,
            "hueMin": self.hue_min,
            "hueMax": self.hue_max,
            "hueMin2": self.hue_min2,
            "hueMax2": self.hue_max2,
            "sMin": self.s_min,
            "sMax": self.s_max,
            "vMin": self.v_min,
            "vMax": self.v_max,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ClothingHint:
        return cls(
            name=str(payload["name"]),
            label=str(payload.get("label") or payload["name"]),
            hue_min=int(payload["hueMin"]),
            hue_max=int(payload["hueMax"]),
            hue_min2=_optional_int(payload.get("hueMin2")),
            hue_max2=_optional_int(payload.get("hueMax2")),
            s_min=int(payload["sMin"]),
            s_max=_optional_int(payload.get("sMax")),
            v_min=int(payload["vMin"]),
            v_max=_optional_int(payload.get("vMax")),
        )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def normalize_color_name(value: str) -> str:
    key = tr_lower(value).strip()
    if key not in _COLOR_ALIASES:
        known = "sarı, kırmızı, mavi, yeşil, siyah, beyaz"
        raise AppError(f"Kıyafet rengi tanınmadı: {value}. Seçenekler: {known}.")
    return _COLOR_ALIASES[key]


def hint_for(value: str) -> ClothingHint | None:
    """Return a hue range. ``yok`` clears the hint."""
    name = normalize_color_name(value)
    if not name:
        return None
    table = {
        "sari": ClothingHint("sari", "sarı", 10, 25, None, None, 80, None, 80, None),
        "kirmizi": ClothingHint("kirmizi", "kırmızı", 0, 8, 170, 179, 80, None, 80, None),
        "mavi": ClothingHint("mavi", "mavi", 100, 130, None, None, 80, None, 80, None),
        "yesil": ClothingHint("yesil", "yeşil", 40, 80, None, None, 80, None, 80, None),
        "siyah": ClothingHint("siyah", "siyah", 0, 179, None, None, 0, 255, 0, 50),
        "beyaz": ClothingHint("beyaz", "beyaz", 0, 179, None, None, 0, 40, 200, None),
    }
    return table[name]


def torso_bounds(
    x: float,
    y: float,
    width: float,
    height: float,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """Region directly under the chin: 1.2× face width, 2× face height, clipped."""
    torso_w = width * 1.2
    torso_h = height * 2.0
    center_x = x + width / 2.0
    x1 = int(round(center_x - torso_w / 2.0))
    y1 = int(round(y + height))
    x2 = int(round(center_x + torso_w / 2.0))
    y2 = int(round(y1 + torso_h))
    x1 = max(0, min(frame_w, x1))
    x2 = max(0, min(frame_w, x2))
    y1 = max(0, min(frame_h, y1))
    y2 = max(0, min(frame_h, y2))
    return x1, y1, x2, y2


def _mask(hsv: np.ndarray, hint: ClothingHint) -> np.ndarray:
    s_max = 255 if hint.s_max is None else hint.s_max
    v_max = 255 if hint.v_max is None else hint.v_max

    def band(hue_min: int, hue_max: int) -> np.ndarray:
        lower = np.array([hue_min, hint.s_min, hint.v_min], dtype=np.uint8)
        upper = np.array([hue_max, s_max, v_max], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    mask = band(hint.hue_min, hint.hue_max)
    if hint.hue_min2 is not None and hint.hue_max2 is not None:
        mask = cv2.bitwise_or(mask, band(hint.hue_min2, hint.hue_max2))
    return mask


def fraction_in_hint(bgr: np.ndarray, hint: ClothingHint) -> float:
    if bgr.size == 0 or bgr.shape[0] == 0 or bgr.shape[1] == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = _mask(hsv, hint)
    total = float(mask.shape[0] * mask.shape[1])
    return float(np.count_nonzero(mask)) / total


def clothing_matches(
    image: np.ndarray,
    face_xywh: tuple[float, float, float, float],
    hint: ClothingHint,
    *,
    minimum: float = MIN_CLOTHING_FRACTION,
) -> bool:
    frame_h, frame_w = image.shape[:2]
    x1, y1, x2, y2 = torso_bounds(*face_xywh, frame_w, frame_h)
    if x2 <= x1 or y2 <= y1:
        return False
    return fraction_in_hint(image[y1:y2, x1:x2], hint) >= minimum
