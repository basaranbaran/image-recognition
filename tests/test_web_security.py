"""Only this machine's browser, only from the panel's page, never inside a frame."""

import pytest
from fastapi.testclient import TestClient

from recognize.web import app


@pytest.mark.parametrize(
    ("host", "status"),
    [
        ("127.0.0.1:8765", 200),
        ("localhost:8765", 200),
        ("[::1]:8765", 200),
        ("evil.example:8765", 400),
        ("0.0.0.0:8765", 400),
        ("127.0.0.1.nip.io", 400),
        ("localhost.evil.example", 400),
        ("user@127.0.0.1", 400),
        (None, 200),
    ],
)
def test_only_local_host_names_are_served(home, host: str | None, status: int) -> None:
    with TestClient(app) as client:
        headers = {} if host is None else {"Host": host}
        response = client.get("/api/ayarlar", headers=headers)
    assert response.status_code == status
    if status == 400:
        assert response.json()["detail"] == "Bu panel yalnızca http://127.0.0.1 adresinden açılır."


def test_cross_site_api_reads_are_refused_but_the_page_opens(home) -> None:
    with TestClient(app) as client:
        refused = client.get("/api/kisiler", headers={"Sec-Fetch-Site": "cross-site"})
        assert refused.status_code == 403
        cover = client.get(
            "/api/kisiler/kimse/kapak.jpg", headers={"Sec-Fetch-Site": "cross-site"}
        )
        assert cover.status_code == 403
        assert client.get("/", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 200
        for value in ("same-origin", "none"):
            assert client.get("/api/kisiler", headers={"Sec-Fetch-Site": value}).status_code == 200
        assert client.get("/api/kisiler").status_code == 200


def test_pages_cannot_be_framed(home) -> None:
    with TestClient(app) as client:
        for path in ("/", "/kamera", "/api/ayarlar"):
            response = client.get(path)
            assert response.status_code == 200, path
            assert response.headers["x-frame-options"] == "DENY"
            assert response.headers["content-security-policy"] == "frame-ancestors 'none'"
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.headers["referrer-policy"] == "no-referrer"
            assert response.headers["cross-origin-resource-policy"] == "same-origin"
        assert client.get("/").headers["cache-control"] == "no-cache"
        assert client.get("/static/app.css").headers["cache-control"] == "no-cache"


def test_favicon_is_served(home) -> None:
    with TestClient(app) as client:
        response = client.get("/favicon.ico")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert "max-age=86400" in response.headers["cache-control"]
