""" "Bu kişi değil": a copy moves aside inside the output, the report follows, undo restores."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import scanned_output
from recognize import results, web
from recognize.web import app


def _tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _report(output: Path) -> dict:
    return json.loads((output / "rapor.json").read_text(encoding="utf-8"))


def _reject(client: TestClient, output: Path, path: str, slug: str | None = None):
    body = {"output": str(output), "path": path}
    if slug:
        body["slug"] = slug
    return client.post("/api/sonuclar/yanlis-eslesme", json=body)


def _undo(client: TestClient, output: Path, path: str):
    return client.post("/api/sonuclar/geri-al", json={"output": str(output), "path": path})


def test_wrong_match_moves_the_copy_aside_and_keeps_the_source(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    source_before = _tree(tmp_path / "etkinlik")
    with TestClient(app) as client:
        response = _reject(client, output, "Emre Kaya/ikisi.jpg", "emre")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["movedTo"] == "_Yanlış eşleşmeler/Emre Kaya/ikisi.jpg"
    assert body["already"] is False
    assert body["person"] == {"slug": "emre", "files": 1, "rejected": 1}
    assert (output / "_Yanlış eşleşmeler" / "Emre Kaya" / "ikisi.jpg").is_file()
    assert not (output / "Emre Kaya" / "ikisi.jpg").exists()
    assert (output / "Ali Veli" / "ikisi.jpg").is_file()
    assert _tree(tmp_path / "etkinlik") == source_before
    report = _report(output)
    record = next(r for r in report["results"] if r["source"] == "ikisi.jpg")
    assert record["copiedTo"] == ["Ali Veli/ikisi.jpg"]
    assert [m["slug"] for m in record["matches"]] == ["ali"]
    assert record["rejected"][0]["slug"] == "emre"
    assert record["rejected"][0]["path"] == "Emre Kaya/ikisi.jpg"
    emre = next(p for p in report["summary"]["people"] if p["slug"] == "emre")
    assert emre["files"] == 1 and emre["rejected"] == 1
    assert report["summary"]["rejected"] == 1
    assert report["corrections"][0]["action"] == "reject"
    with TestClient(app) as client:
        listing = client.get("/api/sonuclar", params={"output": str(output)}).json()
    rejected = next(p for p in listing["people"] if p["slug"] == "emre")["rejected"]
    assert rejected[0]["path"] == "_Yanlış eşleşmeler/Emre Kaya/ikisi.jpg"
    assert rejected[0]["thumb"] is not None


def test_undo_restores_file_and_report(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    original = _report(output)
    with TestClient(app) as client:
        moved = _reject(client, output, "Emre Kaya/ikisi.jpg").json()["movedTo"]
        response = _undo(client, output, moved)
    assert response.status_code == 200, response.text
    assert response.json()["path"] == "Emre Kaya/ikisi.jpg"
    assert (output / "Emre Kaya" / "ikisi.jpg").is_file()
    assert not (output / "_Yanlış eşleşmeler" / "Emre Kaya" / "ikisi.jpg").exists()
    restored = _report(output)
    corrections = restored.pop("corrections")
    assert [row["action"] for row in corrections] == ["reject", "undo"]
    restored.pop("scannedAt")
    original.pop("scannedAt")
    record = next(r for r in restored["results"] if r["source"] == "ikisi.jpg")
    assert record["rejected"] == []
    record.pop("rejected")
    assert sorted(m["slug"] for m in record["matches"]) == ["ali", "emre"]
    assert sorted(record["copiedTo"]) == ["Ali Veli/ikisi.jpg", "Emre Kaya/ikisi.jpg"]
    assert restored["summary"] == original["summary"]


def test_rejecting_on_a_move_scan_keeps_the_only_copy(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path, move=True)
    count_before = len(_tree(tmp_path))
    with TestClient(app) as client:
        response = _reject(client, output, "Emre Kaya/emre.jpg")
    assert response.status_code == 200
    copies = [p for p in output.rglob("emre.jpg")]
    assert len(copies) == 1
    assert copies[0] == output / "_Yanlış eşleşmeler" / "Emre Kaya" / "emre.jpg"
    assert len(_tree(tmp_path)) == count_before


def test_reject_twice_is_idempotent_and_collisions_get_a_suffix(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    with TestClient(app) as client:
        first = _reject(client, output, "Emre Kaya/ikisi.jpg").json()
        second = _reject(client, output, "Emre Kaya/ikisi.jpg").json()
        assert second["already"] is True and second["movedTo"] == first["movedTo"]
        # A rescan re-copies the file; rejecting it again must not overwrite the first reject.
        (output / "Emre Kaya" / "ikisi.jpg").write_bytes(b"yeniden")
        report = _report(output)
        record = next(r for r in report["results"] if r["source"] == "ikisi.jpg")
        record["copiedTo"].append("Emre Kaya/ikisi.jpg")
        record["matches"].append({"name": "Emre Kaya", "slug": "emre", "score": 0.5})
        (output / "rapor.json").write_text(json.dumps(report), encoding="utf-8")
        third = _reject(client, output, "Emre Kaya/ikisi.jpg").json()
    assert third["movedTo"] == "_Yanlış eşleşmeler/Emre Kaya/ikisi (2).jpg"
    assert (output / third["movedTo"]).read_bytes() == b"yeniden"


def test_corrections_refused_while_a_scan_writes_there(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    with web._scan_lock:
        web._scan.update(state="running", destination=str(output.resolve()))
    try:
        with TestClient(app) as client:
            response = _reject(client, output, "Emre Kaya/ikisi.jpg")
            listing = client.get("/api/sonuclar", params={"output": str(output)}).json()
    finally:
        with web._scan_lock:
            web._scan.update(state="idle", destination=None)
    assert response.status_code == 409
    assert response.json()["detail"] == "Bu klasöre tarama sürüyor. Bitince yeniden deneyin."
    assert listing["scanRunning"] is True
    assert (output / "Emre Kaya" / "ikisi.jpg").is_file()


def test_undo_refuses_to_overwrite(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    with TestClient(app) as client:
        moved = _reject(client, output, "Emre Kaya/ikisi.jpg").json()["movedTo"]
        (output / "Emre Kaya" / "ikisi.jpg").write_bytes(b"yeni kopya")
        response = _undo(client, output, moved)
    assert response.status_code == 409
    assert response.json()["detail"] == "Kişi klasöründe aynı adlı bir dosya var; geri alınamadı."
    assert (output / "Emre Kaya" / "ikisi.jpg").read_bytes() == b"yeni kopya"
    assert (output / moved).is_file()


def test_report_write_failure_rolls_back_the_move(
    home, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = scanned_output(tmp_path)
    before = _report(output)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(results, "write_json_atomic", boom)
    with TestClient(app) as client:
        response = _reject(client, output, "Emre Kaya/ikisi.jpg")
    assert response.status_code == 500
    assert response.json()["detail"] == "Rapor güncellenemedi; dosya yerine geri kondu."
    assert (output / "Emre Kaya" / "ikisi.jpg").is_file()
    assert not (output / "_Yanlış eşleşmeler").exists() or not any(
        (output / "_Yanlış eşleşmeler").rglob("*.jpg")
    )
    assert _report(output) == before


def test_reject_messages_for_unknown_wrong_owner_and_missing_file(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    (output / "Emre Kaya" / "emre.jpg").unlink()
    with TestClient(app) as client:
        unknown = _reject(client, output, "Emre Kaya/yok.jpg")
        wrong = _reject(client, output, "Emre Kaya/ikisi.jpg", "ali")
        gone = _reject(client, output, "Emre Kaya/emre.jpg")
        undo = _undo(client, output, "_Yanlış eşleşmeler/Emre Kaya/yok.jpg")
    assert unknown.status_code == 404
    assert unknown.json()["detail"] == "Bu dosya tarama raporunda yok."
    assert wrong.status_code == 400
    assert wrong.json()["detail"] == "Dosya bu kişinin klasöründe değil."
    assert gone.status_code == 404
    assert gone.json()["detail"] == (
        "Dosya kişi klasöründe yok; elle taşınmış ya da silinmiş olabilir."
    )
    assert (undo.status_code, undo.json()["detail"]) == (404, "Geri alınacak bir ayıklama yok.")
