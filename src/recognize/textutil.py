"""Turkish case folding, slugs, and folder names."""

from __future__ import annotations

import re
import unicodedata

from recognize.errors import AppError

_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})
_TR_UPPER = str.maketrans({"i": "İ", "ı": "I"})
_SLUG = str.maketrans(
    {
        "ç": "c",
        "Ç": "c",
        "ğ": "g",
        "Ğ": "g",
        "ı": "i",
        "I": "i",
        "İ": "i",
        "i": "i",
        "ö": "o",
        "Ö": "o",
        "ş": "s",
        "Ş": "s",
        "ü": "u",
        "Ü": "u",
    }
)

_MAX_NAME = 80


def tr_lower(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(_TR_LOWER).lower()


def tr_upper(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(_TR_UPPER).upper()


def slugify(name: str) -> str:
    text = unicodedata.normalize("NFC", name).translate(_SLUG).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "kisi"


def clean_person_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", unicodedata.normalize("NFC", name)).strip()
    if not cleaned:
        raise AppError("Kişi adı boş olamaz.")
    if len(cleaned) > _MAX_NAME:
        raise AppError("Kişi adı 80 karakterden uzun olamaz.")
    if any(char in cleaned for char in "/\\\x00"):
        raise AppError("Kişi adında klasör ayıracı olamaz.")
    return cleaned


_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{kind}{n}" for kind in ("COM", "LPT") for n in range(1, 10)
}


def folder_name(name: str) -> str:
    """The person's folder name; also valid on Windows and on exFAT USB drives."""
    cleaned = re.sub(r'[<>:"|?*\x00-\x1f]', " ", clean_person_name(name))
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip(". ")
    if cleaned in {"", ".", ".."}:
        raise AppError("Kişi adı klasör adı olamıyor.")
    if cleaned.split(".")[0].upper() in _RESERVED:
        cleaned = f"{cleaned}_"
    return cleaned
