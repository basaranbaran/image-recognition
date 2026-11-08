"""A person deleted in the panel leaves the archive, photos included, until they are restored."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from conftest import scanned_output
from recognize.web import app


def _results(client: TestClient, output: Path, **params: str) -> dict:
    return client.get("/api/sonuclar", params={"output": str(output), **params}).json()


def test_deleted_person_leaves_the_archive_with_their_photos(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    with TestClient(app) as client:
        assert _results(client, output)["summary"]["matchedFiles"] == 2
        deleted = client.delete("/api/kisiler/emre")
        assert deleted.status_code == 200, deleted.text
        after = _results(client, output)
        photo = client.get(
            "/api/sonuclar/onizleme", params={"output": str(output), "path": "Emre Kaya/emre.jpg"}
        )
        sheet = client.get("/api/sonuclar", params={"output": str(output), "person": "emre"})
    assert [row["slug"] for row in after["people"]] == ["ali"]
    assert [row["name"] for row in after["summary"]["people"]] == ["Ali Veli"]
    assert after["summary"]["matchedFiles"] == 1
    assert "Emre Kaya/" not in json.dumps(after)
    assert "emre" not in json.dumps(after["people"])
    assert photo.status_code == 404
    assert sheet.status_code == 404


def test_restoring_brings_the_person_back(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    with TestClient(app) as client:
        token = client.delete("/api/kisiler/emre").json()["restoreToken"]
        restored = client.post("/api/kisiler/geri-al", json={"token": token})
        assert restored.status_code == 200, restored.text
        body = _results(client, output)
    assert [row["slug"] for row in body["people"]] == ["ali", "emre"]
    assert body["summary"]["matchedFiles"] == 2


def test_move_aside_skips_a_deleted_person(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    (output / "Ali Veli" / "eski.jpg").write_bytes(b"old")
    (output / "Emre Kaya" / "eski.jpg").write_bytes(b"old")
    with TestClient(app) as client:
        client.delete("/api/kisiler/emre")
        response = client.post("/api/sonuclar/eskileri-ayir", json={"output": str(output)})
    assert response.json() == {"moved": 1}
    assert not (output / "Ali Veli" / "eski.jpg").exists()
    assert (output / "Emre Kaya" / "eski.jpg").read_bytes() == b"old"
