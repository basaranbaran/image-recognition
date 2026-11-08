"""Turkish names and slugs."""

import pytest

from recognize.errors import AppError
from recognize.textutil import clean_person_name, folder_name, slugify, tr_lower, tr_upper


def test_turkish_case_and_slug() -> None:
    assert tr_lower("IŞIK") == "ışık"
    assert tr_lower("İSTANBUL") == "istanbul"
    assert tr_lower("ı") == "ı"
    assert tr_upper("ışık") == "IŞIK"
    assert tr_upper("istanbul") == "İSTANBUL"
    assert slugify("Işık Yılmaz") == "isik-yilmaz"
    assert slugify("İpek Şahin") == "ipek-sahin"
    assert folder_name("Ayşe Yılmaz") == "Ayşe Yılmaz"


def test_empty_name_rejected() -> None:
    with pytest.raises(AppError):
        clean_person_name("   ")


def test_long_name_rejected() -> None:
    with pytest.raises(AppError):
        clean_person_name("A" * 81)
