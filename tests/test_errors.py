"""Every failure reaches the page as a Turkish sentence in ``detail``."""

import pytest
from fastapi.testclient import TestClient

from recognize import web


def test_validation_errors_have_a_string_detail(home) -> None:
    with TestClient(web.app) as client:
        response = client.post("/api/tara", json={"source": 3})
    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "İstek eksik ya da hatalı. Sayfayı yenileyip yeniden deneyin."
    assert "destination" in body["fields"]


def test_unexpected_errors_are_json(home, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom():
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(web, "load_people_safe", boom)
    with TestClient(web.app, raise_server_exceptions=False) as client:
        response = client.get("/api/kisiler")
    assert response.status_code == 500
    assert response.json() == {
        "detail": "Beklenmeyen bir hata oldu. Ayrıntı panelin çalıştığı terminalde."
    }
