"""Deleting a person is a move into the app's bin, undone with a token; broken folders too."""

import json
import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from recognize import web
from recognize.gallery import add_references, load_people, load_people_safe, purge_deleted


def _enroll(name: str, rows: int = 1) -> str:
    samples = [
        (np.array([1.0, float(i), 0.0, 0.0], dtype=np.float32), 0.9, b"jpeg", f"{i}.jpg")
        for i in range(rows)
    ]
    person, _created = add_references(name, samples, None, update_clothing=False)
    return person.slug


def test_soft_delete_and_restore(home) -> None:
    slug = _enroll("Baran Başaran")
    with TestClient(web.app) as client:
        before = client.get("/api/kisiler").json()["people"][0]
        assert before["createdAt"]
        assert "?v=1-" in before["cover"]
        removed = client.delete(f"/api/kisiler/{slug}")
        assert removed.status_code == 200, removed.text
        body = removed.json()
        assert body["deleted"] == slug and body["name"] == "Baran Başaran"
        token = body["restoreToken"]
        assert token.startswith(f"{slug}-")
        assert client.get("/api/kisiler").json()["people"] == []
        assert (home / "veri" / "silinenler" / token / "person.json").is_file()
        restored = client.post("/api/kisiler/geri-al", json={"token": token})
        assert restored.status_code == 200, restored.text
        assert restored.json() == {"restored": slug, "name": "Baran Başaran"}
        assert [p["slug"] for p in client.get("/api/kisiler").json()["people"]] == [slug]
        again = client.post("/api/kisiler/geri-al", json={"token": token})
        assert again.status_code == 404
        bad = client.post("/api/kisiler/geri-al", json={"token": "../x"})
        assert bad.status_code == 400


def test_restore_refuses_when_the_slug_is_taken(home) -> None:
    slug = _enroll("Emre")
    with TestClient(web.app) as client:
        token = client.delete(f"/api/kisiler/{slug}").json()["restoreToken"]
        _enroll("Emre")
        response = client.post("/api/kisiler/geri-al", json={"token": token})
    assert response.status_code == 409
    assert response.json()["detail"] == "Bu adla yeni bir kayıt var; geri alınamadı."


def test_delete_refused_during_a_scan(home) -> None:
    slug = _enroll("Emre")
    with web._scan_lock:
        web._scan["state"] = "running"
    try:
        with TestClient(web.app) as client:
            response = client.delete(f"/api/kisiler/{slug}")
    finally:
        with web._scan_lock:
            web._scan["state"] = "idle"
    assert response.status_code == 409
    assert (home / "veri" / "kisiler" / slug).is_dir()


def test_purge_removes_old_entries_at_startup(home) -> None:
    slug = _enroll("Emre")
    with TestClient(web.app) as client:
        token = client.delete(f"/api/kisiler/{slug}").json()["restoreToken"]
    folder = home / "veri" / "silinenler" / token
    old = time.time() - 8 * 86400
    os.utime(folder, (old, old))
    assert purge_deleted() == 1
    assert not folder.exists()
    _enroll("Ali")
    with TestClient(web.app) as client:
        token = client.delete("/api/kisiler/ali").json()["restoreToken"]
        fresh = home / "veri" / "silinenler" / token
        old = time.time() - 8 * 86400
        os.utime(fresh, (old, old))
    with TestClient(web.app):
        pass
    assert not fresh.exists()


def test_people_list_survives_a_broken_record(home) -> None:
    _enroll("Emre")
    broken = home / "veri" / "kisiler" / "bozuk"
    broken.mkdir(parents=True)
    (broken / "person.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_people()
    people, problems = load_people_safe()
    assert [p.slug for p in people] == ["emre"]
    assert problems[0]["folder"] == "bozuk" and "silip yeniden tanıtın" in problems[0]["detail"]
    with TestClient(web.app) as client:
        body = client.get("/api/kisiler").json()
        assert [p["slug"] for p in body["people"]] == ["emre"]
        assert body["problems"][0]["folder"] == "bozuk"
        source = home.parent / "etkinlik"
        source.mkdir()
        refused = client.post(
            "/api/tara",
            json={"source": str(source), "destination": str(home / "x"), "people": ["bozuk"]},
        )
        assert refused.status_code == 400
        assert refused.json()["detail"] == (
            "Seçilen kişilerden birinin kaydı bozuk. Kaydı silip yeniden tanıtın."
        )
        deleted = client.delete("/api/kisiler/bozuk")
    assert deleted.status_code == 200
    assert deleted.json()["name"] == "bozuk"
    assert not broken.exists()


def test_mismatched_rows_recover_the_common_prefix(home) -> None:
    slug = _enroll("Emre", rows=3)
    folder = home / "veri" / "kisiler" / slug
    payload = json.loads((folder / "person.json").read_text(encoding="utf-8"))
    payload["references"] = payload["references"][:2]
    (folder / "person.json").write_text(json.dumps(payload), encoding="utf-8")
    people, problems = load_people_safe()
    assert problems == []
    assert people[0].embeddings.shape == (2, 4) and len(people[0].references) == 2
