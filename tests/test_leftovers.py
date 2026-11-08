"""Files an earlier run left in a person folder: listed, and moved aside on request."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from conftest import scanned_output
from recognize.web import app

ASIDE = "_Önceki taramalar"


def _listing(client: TestClient, output: Path) -> dict:
    return client.get("/api/sonuclar", params={"output": str(output)}).json()


def _person(body: dict, slug: str) -> dict:
    return next(row for row in body["people"] if row["slug"] == slug)


def _aside_request(client: TestClient, output: Path, slug: str | None = None):
    body = {"output": str(output)}
    if slug is not None:
        body["slug"] = slug
    return client.post("/api/sonuclar/eskileri-ayir", json=body)


def _mark_unfinished(output: Path) -> None:
    report = json.loads((output / "rapor.json").read_text(encoding="utf-8"))
    report["complete"] = False
    (output / "rapor.json").write_text(json.dumps(report), encoding="utf-8")


def test_leftovers_are_listed_and_only_media_counts(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    folder = output / "Ali Veli"
    (folder / "eski.jpg").write_bytes(b"old")
    (folder / "eski-video.mp4").write_bytes(b"old video")
    (folder / "notlar.txt").write_text("not media", encoding="utf-8")
    (folder / ".DS_Store").write_bytes(b"finder")
    with TestClient(app) as client:
        body = _listing(client, output)
    ali = _person(body, "ali")
    assert ali["leftovers"] == ["eski-video.mp4", "eski.jpg"]
    assert ali["counts"]["leftovers"] == 2
    assert body["summary"]["leftovers"] == 2
    assert _person(body, "emre")["leftovers"] == []


def test_leftovers_stay_hidden_until_the_scan_finishes(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    (output / "Ali Veli" / "eski.jpg").write_bytes(b"old")
    _mark_unfinished(output)
    with TestClient(app) as client:
        body = _listing(client, output)
    assert _person(body, "ali")["leftovers"] == []
    assert body["summary"]["leftovers"] == 0


def test_move_aside_keeps_the_file_and_its_folder(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    old = output / "Ali Veli" / "eski.jpg"
    old.write_bytes(b"old")
    with TestClient(app) as client:
        response = _aside_request(client, output, "ali")
        after = _listing(client, output)
    assert response.status_code == 200, response.text
    assert response.json() == {"moved": 1}
    assert not old.exists()
    assert (output / ASIDE / "Ali Veli" / "eski.jpg").read_bytes() == b"old"
    assert (output / "Ali Veli" / "ikisi.jpg").is_file()
    assert _person(after, "ali")["leftovers"] == []


def test_move_aside_never_overwrites_an_earlier_aside_file(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    (output / "Ali Veli" / "eski.jpg").write_bytes(b"second")
    aside = output / ASIDE / "Ali Veli"
    aside.mkdir(parents=True)
    (aside / "eski.jpg").write_bytes(b"first")
    with TestClient(app) as client:
        response = _aside_request(client, output)
    assert response.json() == {"moved": 1}
    assert (aside / "eski.jpg").read_bytes() == b"first"
    assert (aside / "eski (2).jpg").read_bytes() == b"second"


def test_move_aside_refuses_an_unfinished_scan(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    (output / "Ali Veli" / "eski.jpg").write_bytes(b"old")
    _mark_unfinished(output)
    with TestClient(app) as client:
        response = _aside_request(client, output)
    assert response.status_code == 409
    assert (output / "Ali Veli" / "eski.jpg").is_file()


def test_move_aside_unknown_person_is_not_found(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    with TestClient(app) as client:
        response = _aside_request(client, output, "kimse")
    assert response.status_code == 404
