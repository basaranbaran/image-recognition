"""The results listing and its thumbnails read only what the report lists."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import jpeg_bytes, scanned_output
from recognize import thumbs
from recognize.store import remember_scan
from recognize.web import app


def _results(client: TestClient, output: Path, **params: str):
    return client.get("/api/sonuclar", params={"output": str(output), **params})


def test_results_group_report_copies_by_person(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    with TestClient(app) as client:
        response = _results(client, output)
    assert response.status_code == 200
    body = response.json()
    assert body["complete"] is True and body["relocated"] is False
    assert body["sourceExists"] is True and body["scanRunning"] is False
    people = {row["slug"]: row for row in body["people"]}
    assert [row["slug"] for row in body["people"]] == ["ali", "emre"]
    emre = people["emre"]
    assert [item["path"] for item in emre["files"]] == [
        "Emre Kaya/emre.jpg",
        "Emre Kaya/ikisi.jpg",
    ]
    both = emre["files"][1]
    assert both["smallFace"] is True
    assert "smallFace" in both["review"]
    assert both["with"] == ["ali"]
    assert both["exists"] is True and both["bytes"] > 0
    assert both["thumb"].startswith("/api/sonuclar/onizleme?") and "v=" in both["thumb"]
    assert emre["counts"]["files"] == 2 and emre["counts"]["review"] == 1
    assert emre["enrolled"] is False and emre["cover"] is None
    ali = people["ali"]
    assert ali["files"][0]["box"] == [0.1, 0.1, 0.2, 0.3]
    assert ali["folderExists"] is True
    assert body["summary"]["missing"] == 0
    assert body["unmatched"] == {"errors": [], "skipped": [], "uncertain": []}


def test_results_default_output_is_the_last_scan(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    remember_scan(output, {"files": 2, "matchedFiles": 2}, "2026-11-06T00:00:00+00:00")
    with TestClient(app) as client:
        explicit = _results(client, output).json()
        default = client.get("/api/sonuclar").json()
        person = _results(client, output, person="ali").json()
        missing = _results(client, output, person="kimse")
    assert default == explicit
    assert [row["slug"] for row in person["people"]] == ["ali"]
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Bu kişi bu taramada yok."


def test_results_refuse_relative_missing_and_corrupt_reports(home, tmp_path: Path) -> None:
    empty = tmp_path / "bos"
    empty.mkdir()
    broken = tmp_path / "bozuk"
    broken.mkdir()
    (broken / "rapor.json").write_text("{not json", encoding="utf-8")
    shaped = tmp_path / "sekil"
    shaped.mkdir()
    (shaped / "rapor.json").write_text('{"summary": {}}', encoding="utf-8")
    with TestClient(app) as client:
        cases = [
            ("cikti", 400, "Çıktı klasörünün tam yolunu verin."),
            (str(tmp_path / "yok"), 404, "Çıktı klasörü bulunamadı."),
            (str(empty), 404, "Bu klasörde tarama raporu yok. Önce ayırmayı başlatın."),
            (str(broken), 422, "Tarama raporu okunamadı. rapor.json bozuk olabilir."),
            (str(shaped), 422, "Bu rapor.json bir tarama raporu değil."),
        ]
        for output, status, detail in cases:
            response = client.get("/api/sonuclar", params={"output": output})
            assert (response.status_code, response.json()["detail"]) == (status, detail), output


def test_results_read_an_old_report_without_new_fields(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    report = json.loads((output / "rapor.json").read_text(encoding="utf-8"))
    for key in ("complete", "progress"):
        report.pop(key, None)
    for record in report["results"]:
        record.pop("duplicateOf", None)
        record.pop("skipped", None)
        for match in record["matches"]:
            for key in ("learned", "facePx", "smallFace", "box", "samples"):
                match.pop(key, None)
    (output / "rapor.json").write_text(json.dumps(report), encoding="utf-8")
    with TestClient(app) as client:
        body = _results(client, output).json()
    assert body["complete"] is True
    item = body["people"][0]["files"][0]
    assert item["facePx"] is None and item["samples"] is None and item["box"] is None
    assert item["smallFace"] is False and item["learned"] is False


def test_results_after_a_move_scan_never_point_at_the_source(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path, move=True)
    assert not any((tmp_path / "etkinlik").iterdir())
    with TestClient(app) as client:
        body = _results(client, output).json()
        assert body["move"] is True and body["sourceExists"] is True
        for row in body["people"]:
            for item in row["files"]:
                assert "path=" in item["thumb"]
                assert (output / item["path"]).is_file()
                assert client.get(item["thumb"]).status_code == 200


def test_missing_copy_is_marked_not_hidden(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    (output / "Emre Kaya" / "emre.jpg").unlink()
    with TestClient(app) as client:
        body = _results(client, output).json()
        gone = next(
            item for row in body["people"] for item in row["files"] if item["name"] == "emre.jpg"
        )
        assert gone["exists"] is False and gone["thumb"] is None
        assert body["summary"]["missing"] == 1
        refused = client.get(
            "/api/sonuclar/onizleme", params={"output": str(output), "path": gone["path"]}
        )
    assert refused.status_code == 404
    assert refused.json()["detail"] == (
        "Dosya artık yerinde değil. Taşınmış ya da silinmiş olabilir."
    )


def test_relocated_output_is_still_browsable(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    moved = tmp_path / "usb" / "cikti"
    shutil.copytree(output, moved)
    with TestClient(app) as client:
        body = _results(client, moved).json()
        assert body["relocated"] is True
        thumb = body["people"][0]["files"][0]["thumb"]
        assert f"output={str(moved).replace('/', '%2F')}" in thumb
        assert client.get(thumb).status_code == 200


def test_thumbnail_is_an_upright_jpeg_with_cache_headers(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    with TestClient(app) as client:
        body = _results(client, output).json()
        item = next(
            item for row in body["people"] for item in row["files"] if item["name"] == "ikisi.jpg"
        )
        response = client.get(item["thumb"])
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert "immutable" in response.headers["cache-control"]
        image = Image.open(__import__("io").BytesIO(response.content))
        assert image.size == (48, 64)
        again = client.get(item["thumb"], headers={"If-None-Match": response.headers["etag"]})
        assert again.status_code == 304
        stale = client.get(item["thumb"].rsplit("&v=", 1)[0] + "&v=eski")
        assert stale.headers["cache-control"] == "private, no-cache"
        big = client.get(item["thumb"] + "&size=1280")
        assert Image.open(__import__("io").BytesIO(big.content)).size == (48, 64)


@pytest.mark.parametrize(
    "path",
    [
        "../etkinlik/emre.jpg",
        "/etc/hosts",
        "rapor.json",
        "Emre Kaya/../../etkinlik/emre.jpg",
        "Emre Kaya/eklenmis.jpg",
        "",
        "Emre Kaya\\..\\..\\etkinlik\\emre.jpg",
    ],
)
def test_thumbnail_serves_only_files_the_report_lists(home, tmp_path: Path, path: str) -> None:
    output = scanned_output(tmp_path)
    (output / "Emre Kaya" / "eklenmis.jpg").write_bytes(jpeg_bytes())
    with TestClient(app) as client:
        response = client.get(
            "/api/sonuclar/onizleme", params={"output": str(output), "path": path}
        )
    assert response.status_code == 404
    assert response.json()["detail"] == "Bu dosya tarama raporunda yok."


def test_thumbnail_refuses_a_symlink_that_escapes(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    link = output / "Emre Kaya" / "kisayol.jpg"
    link.symlink_to(tmp_path / "etkinlik" / "emre.jpg")
    report = json.loads((output / "rapor.json").read_text(encoding="utf-8"))
    report["results"][0]["copiedTo"].append("Emre Kaya/kisayol.jpg")
    (output / "rapor.json").write_text(json.dumps(report), encoding="utf-8")
    with TestClient(app) as client:
        response = client.get(
            "/api/sonuclar/onizleme",
            params={"output": str(output), "path": "Emre Kaya/kisayol.jpg"},
        )
    assert response.status_code == 404


def test_thumbnail_of_heic_and_transparent_png(home, tmp_path: Path) -> None:
    pillow_heif = pytest.importorskip("pillow_heif")
    output = scanned_output(tmp_path)
    folder = output / "Emre Kaya"
    heif = pillow_heif.from_pillow(Image.new("RGB", (40, 30), (200, 40, 10)))
    heif.save(folder / "tel.heic", quality=90)
    png = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    png.save(folder / "seffaf.png")
    report = json.loads((output / "rapor.json").read_text(encoding="utf-8"))
    report["results"][0]["copiedTo"] += ["Emre Kaya/tel.heic", "Emre Kaya/seffaf.png"]
    (output / "rapor.json").write_text(json.dumps(report), encoding="utf-8")
    with TestClient(app) as client:
        heic = client.get(
            "/api/sonuclar/onizleme", params={"output": str(output), "path": "Emre Kaya/tel.heic"}
        )
        assert heic.status_code == 200
        assert Image.open(__import__("io").BytesIO(heic.content)).size == (40, 30)
        flat = client.get(
            "/api/sonuclar/onizleme",
            params={"output": str(output), "path": "Emre Kaya/seffaf.png"},
        )
    assert flat.status_code == 200
    image = Image.open(__import__("io").BytesIO(flat.content)).convert("RGB")
    assert image.getpixel((10, 10)) == (255, 255, 255)


def test_video_poster_and_header_only_stub(
    home, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cv2

    output = scanned_output(tmp_path)
    folder = output / "Emre Kaya"
    clip = folder / "klip.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for index in range(20):
        writer.write(np.full((48, 64, 3), index * 10, np.uint8))
    writer.release()
    (folder / "yarim.mp4").write_bytes(b"\x00" * 2048)
    report = json.loads((output / "rapor.json").read_text(encoding="utf-8"))
    report["results"][0]["copiedTo"] += ["Emre Kaya/klip.mp4", "Emre Kaya/yarim.mp4"]
    (output / "rapor.json").write_text(json.dumps(report), encoding="utf-8")
    with TestClient(app) as client:
        poster = client.get(
            "/api/sonuclar/onizleme", params={"output": str(output), "path": "Emre Kaya/klip.mp4"}
        )
        assert poster.status_code == 200
        assert poster.headers["content-type"] == "image/jpeg"

        class _Stub:
            def __init__(self, *_args: object) -> None:
                pass

            def isOpened(self) -> bool:
                return True

            def set(self, *_args: object) -> None:
                pass

            def get(self, prop: int) -> float:
                return 850.0 if prop == cv2.CAP_PROP_FRAME_COUNT else 25.0

            def read(self):
                return False, None

            def release(self) -> None:
                pass

        monkeypatch.setattr(thumbs.cv2, "VideoCapture", _Stub)
        stub = client.get(
            "/api/sonuclar/onizleme",
            params={"output": str(output), "path": "Emre Kaya/yarim.mp4"},
        )
    assert stub.status_code == 422
    assert stub.json()["detail"] == "Videodan kare okunamadı. Dosya yarım kopyalanmış olabilir."


def test_thumbnail_size_presets(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    with TestClient(app) as client:
        response = client.get(
            "/api/sonuclar/onizleme",
            params={"output": str(output), "path": "Emre Kaya/emre.jpg", "size": "500"},
        )
    assert response.status_code == 400
    assert response.json()["detail"] == "Önizleme boyutu 320, 640 ya da 1280 olabilir."


def test_thumbnail_cache_renders_once(
    home, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = scanned_output(tmp_path)
    calls: list[Path] = []
    real = thumbs.render

    def counting(path: Path, size: int) -> bytes:
        calls.append(path)
        return real(path, size)

    monkeypatch.setattr(thumbs, "render", counting)
    with TestClient(app) as client:
        for _ in range(2):
            response = client.get(
                "/api/sonuclar/onizleme",
                params={"output": str(output), "path": "Emre Kaya/emre.jpg"},
            )
            assert response.status_code == 200
    assert len(calls) == 1
    cached = list((home / "veri" / "onizleme").rglob("*.jpg"))
    assert len(cached) == 1


def test_history_lists_recent_outputs(home, tmp_path: Path) -> None:
    output = scanned_output(tmp_path)
    remember_scan(output, {"files": 2, "matchedFiles": 2}, "2026-11-06T00:00:00+00:00")
    with TestClient(app) as client:
        items = client.get("/api/sonuclar/gecmis").json()["items"]
    assert items[0]["output"] == str(output.resolve()) and items[0]["exists"] is True
