"""Scanning copies matched files and leaves the rest."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from recognize import media
from recognize.clothing import hint_for
from recognize.errors import AppError
from recognize.gallery import Person
from recognize.match import PersonHit, ScanOptions
from recognize.media import collect_media, directories_overlap
from recognize.scan import InspectionLike, run_scan


class _Fake:
    def __init__(self, hits: dict[str, list[PersonHit]]) -> None:
        self._hits = hits

    def inspect_image(self, path: Path, progress=None) -> InspectionLike:
        del progress
        return InspectionLike(self._hits.get(path.name, []), [])

    def inspect_video(self, path: Path, progress=None) -> InspectionLike:
        del progress
        return InspectionLike(self._hits.get(path.name, []), [])


def _person(root: Path, clothing: bool) -> Person:
    return Person(
        name="Emre",
        slug="emre",
        created_at="",
        clothing=hint_for("sari") if clothing else None,
        embeddings=np.ones((1, 4), dtype=np.float32),
        references=[],
        root=root,
    )


def test_collects_phone_and_camera_formats_but_not_other_files(tmp_path: Path) -> None:
    names = (
        "a.JPG",
        "b.jpeg",
        "c.png",
        "d.webp",
        "e.HEIC",
        "f.heif",
        "g.mp4",
        "h.MOV",
        "i.m4v",
        "j.avi",
        "k.mkv",
    )
    for name in names:
        (tmp_path / name).write_bytes(b"x")
    for name in ("notlar.txt", "kamera.RAF", "ses.m4a", ".hidden.jpg"):
        (tmp_path / name).write_bytes(b"x")
    nested = tmp_path / "__MACOSX"
    nested.mkdir()
    (nested / "skip.jpg").write_bytes(b"s")
    found = {path.name for path in collect_media(tmp_path)}
    assert found == set(names)


def test_heic_photo_decodes_like_a_jpeg(tmp_path: Path) -> None:
    pillow_heif = pytest.importorskip("pillow_heif")
    from PIL import Image

    from recognize.media import load_image

    source = Image.new("RGB", (40, 30), (200, 40, 10))
    heif = pillow_heif.from_pillow(source)
    path = tmp_path / "IMG_0001.HEIC"
    heif.save(path, quality=90)
    image = load_image(path)
    assert image.shape == (30, 40, 3)
    blue, green, red = (int(value) for value in image[15, 20])
    assert red > 150 and blue < 80


def test_mov_video_frames_are_read(tmp_path: Path) -> None:
    import cv2

    from recognize.media import iter_video_frames

    path = tmp_path / "klip.mov"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for index in range(20):
        writer.write(np.full((48, 64, 3), index * 10, np.uint8))
    writer.release()
    frames = list(iter_video_frames(path, 0.5))
    assert len(frames) >= 3
    assert frames[0].shape == (48, 64, 3)


def test_mov_is_scanned_as_a_video(tmp_path: Path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    (source / "IMG_0002.MOV").write_bytes(b"clip")
    (source / "IMG_0002.HEIC").write_bytes(b"photo")
    seen: list[str] = []

    class _Kinds(_Fake):
        def inspect_image(self, path: Path, progress=None) -> InspectionLike:
            seen.append(f"image:{path.name}")
            return InspectionLike([], [])

        def inspect_video(self, path: Path, progress=None) -> InspectionLike:
            seen.append(f"video:{path.name}")
            return InspectionLike([], [])

    report = run_scan(
        source, tmp_path / "out", _Kinds({}), [_person(tmp_path, False)], ScanOptions()
    )
    assert sorted(seen) == ["image:IMG_0002.HEIC", "video:IMG_0002.MOV"]
    assert report["summary"]["videos"] == 1
    assert report["summary"]["images"] == 1


def test_copy_writes_clothing_assisted_and_keeps_source(tmp_path: Path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    photo = source / "kare.jpg"
    photo.write_bytes(b"photo")
    dest = tmp_path / "out"
    hit = PersonHit("emre", "Emre", 0.32, 1, True)
    report = run_scan(
        source,
        dest,
        _Fake({"kare.jpg": [hit]}),
        [_person(tmp_path, True)],
        ScanOptions(move=False),
    )
    assert photo.is_file()
    copied = dest / "Emre" / "kare.jpg"
    assert copied.read_bytes() == b"photo"
    assert report["results"][0]["matches"][0]["clothingAssisted"] is True
    text = (dest / "rapor.json").read_text(encoding="utf-8")
    assert '"clothingAssisted": true' in text


def test_unmatched_file_is_not_copied(tmp_path: Path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    (source / "yellow.jpg").write_bytes(b"yellow-shirt-only")
    dest = tmp_path / "out"
    report = run_scan(
        source,
        dest,
        _Fake({}),
        [_person(tmp_path, True)],
        ScanOptions(),
    )
    assert report["summary"]["matchedFiles"] == 0
    assert not (dest / "Emre").exists()


def test_move_removes_source_after_copy(tmp_path: Path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    photo = source / "kare.jpg"
    photo.write_bytes(b"photo")
    dest = tmp_path / "out"
    hit = PersonHit("emre", "Emre", 0.8, 1, False)
    run_scan(
        source,
        dest,
        _Fake({"kare.jpg": [hit]}),
        [_person(tmp_path, False)],
        ScanOptions(move=True),
    )
    assert not photo.exists()
    assert (dest / "Emre" / "kare.jpg").is_file()


def test_nested_output_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    (source / "a.jpg").write_bytes(b"a")
    with pytest.raises(AppError):
        run_scan(source, source / "out", _Fake({}), [_person(tmp_path, False)], ScanOptions())


def test_overlap_helper() -> None:
    assert directories_overlap(Path("/tmp/a"), Path("/tmp/a/b"))


def test_empty_file_is_skipped_not_an_error(tmp_path: Path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    (source / "yarim.mp4").write_bytes(b"")

    class _Never(_Fake):
        def inspect_video(self, path: Path, progress=None) -> InspectionLike:
            raise AssertionError("an empty file must not be opened")

    report = run_scan(
        source, tmp_path / "out", _Never({}), [_person(tmp_path, False)], ScanOptions()
    )
    record = report["results"][0]
    assert record["error"] is None
    assert "0 bayt" in record["skipped"]
    assert report["summary"]["errors"] == 0
    assert report["summary"]["emptyFiles"] == 1


def test_byte_identical_copy_is_inspected_once_and_written_once(tmp_path: Path) -> None:
    source = tmp_path / "in"
    (source / "FOTOGRAF").mkdir(parents=True)
    (source / "FOTOGRAF 2").mkdir()
    (source / "FOTOGRAF" / "kare.jpg").write_bytes(b"same-bytes")
    (source / "FOTOGRAF 2" / "kare.jpg").write_bytes(b"same-bytes")
    (source / "FOTOGRAF 2" / "baska.jpg").write_bytes(b"other-byte")
    seen: list[str] = []
    hit = PersonHit("emre", "Emre", 0.8, 1, False, 40.0)

    class _Counting(_Fake):
        def inspect_image(self, path: Path, progress=None) -> InspectionLike:
            seen.append(str(path.relative_to(source)))
            return super().inspect_image(path, progress)

    dest = tmp_path / "out"
    report = run_scan(
        source,
        dest,
        _Counting({"kare.jpg": [hit]}),
        [_person(tmp_path, False)],
        ScanOptions(),
    )
    # "FOTOGRAF 2/" sorts before "FOTOGRAF/", so that copy is the one inspected.
    assert sorted(seen) == ["FOTOGRAF 2/baska.jpg", "FOTOGRAF 2/kare.jpg"]
    assert [path.name for path in (dest / "Emre").iterdir()] == ["FOTOGRAF 2__kare.jpg"]
    copy = next(r for r in report["results"] if r["source"] == "FOTOGRAF/kare.jpg")
    assert copy["duplicateOf"] == "FOTOGRAF 2/kare.jpg"
    assert copy["copiedTo"] == []
    assert copy["matches"][0]["facePx"] == 40.0
    summary = report["summary"]
    assert summary["duplicates"] == 1
    assert summary["matchedFiles"] == 1
    assert summary["people"][0]["files"] == 1


def test_move_also_removes_a_copy_of_a_moved_file(tmp_path: Path) -> None:
    source = tmp_path / "in"
    (source / "a").mkdir(parents=True)
    (source / "b").mkdir()
    (source / "a" / "kare.jpg").write_bytes(b"same-bytes")
    (source / "b" / "kare.jpg").write_bytes(b"same-bytes")
    (source / "b" / "kimse.jpg").write_bytes(b"nobody-here")
    hit = PersonHit("emre", "Emre", 0.8, 1, False)
    run_scan(
        source,
        tmp_path / "out",
        _Fake({"kare.jpg": [hit]}),
        [_person(tmp_path, False)],
        ScanOptions(move=True),
    )
    assert not (source / "a" / "kare.jpg").exists()
    assert not (source / "b" / "kare.jpg").exists()
    assert (source / "b" / "kimse.jpg").is_file()


def test_small_matched_face_is_flagged(tmp_path: Path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    (source / "kalabalik.jpg").write_bytes(b"crowd")
    hit = PersonHit("emre", "Emre", 0.55, 1, False, 25.0)
    report = run_scan(
        source,
        tmp_path / "out",
        _Fake({"kalabalik.jpg": [hit]}),
        [_person(tmp_path, False)],
        ScanOptions(),
    )
    assert report["results"][0]["matches"][0]["smallFace"] is True
    assert report["summary"]["people"][0]["smallFaces"] == 1


def test_only_selected_people_get_folders(tmp_path: Path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    (source / "ikisi.jpg").write_bytes(b"both")
    (source / "ali.jpg").write_bytes(b"ali-only")
    emre = PersonHit("emre", "Emre", 0.7, 1, False)
    ali = PersonHit("ali", "Ali", 0.8, 1, False)
    dest = tmp_path / "out"
    report = run_scan(
        source,
        dest,
        _Fake({"ikisi.jpg": [ali, emre], "ali.jpg": [ali]}),
        [_person(tmp_path, False)],
        ScanOptions(),
    )
    assert sorted(path.name for path in dest.iterdir()) == ["Emre", "rapor.json"]
    assert [path.name for path in (dest / "Emre").iterdir()] == ["ikisi.jpg"]
    assert report["summary"]["matchedFiles"] == 1
    assert [row["name"] for row in report["summary"]["people"]] == ["Emre"]
    alone = next(row for row in report["results"] if row["source"] == "ali.jpg")
    assert alone["matches"] == []


def test_second_look_hits_are_written_and_marked_learned(tmp_path: Path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    (source / "net.jpg").write_bytes(b"clear")
    (source / "profil.jpg").write_bytes(b"side")
    first = PersonHit("emre", "Emre", 0.8, 1, False, 90.0)

    class _Learning(_Fake):
        def event_matches(self):
            return {
                source / "net.jpg": [first],
                source / "profil.jpg": [PersonHit("emre", "Emre", 0.62, 1, False, 40.0)],
            }

    dest = tmp_path / "out"
    report = run_scan(
        source,
        dest,
        _Learning({"net.jpg": [first]}),
        [_person(tmp_path, False)],
        ScanOptions(),
    )
    assert sorted(path.name for path in (dest / "Emre").iterdir()) == ["net.jpg", "profil.jpg"]
    side = next(row for row in report["results"] if row["source"] == "profil.jpg")
    assert side["matches"][0]["learned"] is True
    clear = next(row for row in report["results"] if row["source"] == "net.jpg")
    assert len(clear["matches"]) == 1 and clear["matches"][0]["learned"] is False
    row = report["summary"]["people"][0]
    assert row["files"] == 2 and row["learned"] == 1


def test_symlinked_photo_is_copied_under_its_own_name(tmp_path: Path) -> None:
    elsewhere = tmp_path / "baska-yer"
    elsewhere.mkdir()
    (elsewhere / "gercek.jpg").write_bytes(b"photo")
    source = tmp_path / "in"
    source.mkdir()
    (source / "kisayol.jpg").symlink_to(elsewhere / "gercek.jpg")
    hit = PersonHit("emre", "Emre", 0.8, 1, False)
    dest = tmp_path / "out"
    run_scan(source, dest, _Fake({"kisayol.jpg": [hit]}), [_person(tmp_path, False)], ScanOptions())
    assert (dest / "Emre" / "kisayol.jpg").read_bytes() == b"photo"


def test_zero_frame_video_is_flagged_in_the_scan(tmp_path: Path) -> None:
    from recognize.engine import UnreadableMedia

    source = tmp_path / "in"
    source.mkdir()
    (source / "yarim.mp4").write_bytes(b"header only")

    class _Blind(_Fake):
        def inspect_video(self, path: Path, progress=None) -> InspectionLike:
            raise UnreadableMedia(f"Videodan kare okunamadı: {path.name}")

    report = run_scan(
        source, tmp_path / "out", _Blind({}), [_person(tmp_path, False)], ScanOptions()
    )
    record = report["results"][0]
    assert record["error"] is None
    assert record["skipped"] == "Videoda okunabilir kare yok. Kopyalama yarım kalmış olabilir."
    assert report["summary"]["errors"] == 0
    assert report["summary"]["unreadable"] == 1
    assert report["summary"]["emptyFiles"] == 0


def test_hit_carries_a_normalised_box(tmp_path: Path) -> None:
    source = tmp_path / "in"
    source.mkdir()
    (source / "kare.jpg").write_bytes(b"photo")
    hit = PersonHit("emre", "Emre", 0.8, 1, False, 40.0, (0.25, 0.1, 0.2, 0.3))
    report = run_scan(
        source,
        tmp_path / "out",
        _Fake({"kare.jpg": [hit]}),
        [_person(tmp_path, False)],
        ScanOptions(),
    )
    assert report["results"][0]["matches"][0]["box"] == [0.25, 0.1, 0.2, 0.3]
    assert report["complete"] is True and report["progress"] is None


def test_previous_report_is_archived_outside_the_output_and_checkpoints_mark_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from recognize import scan
    from recognize.store import archive_report

    source = tmp_path / "in"
    source.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (source / name).write_bytes(name.encode())
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "rapor.json").write_text(
        '{"scannedAt": "2026-11-05T23:11:03+00:00", "results": []}', encoding="utf-8"
    )
    hit = PersonHit("emre", "Emre", 0.8, 1, False)
    seen_complete: list[bool] = []
    real_write = scan.write_json_atomic

    def spy(path: Path, payload: object) -> None:
        seen_complete.append(bool(payload["complete"]))
        real_write(path, payload)

    monkeypatch.setattr(scan, "write_json_atomic", spy)
    monkeypatch.setattr(scan, "CHECKPOINT_FILES", 1)
    data_root = tmp_path / "veri"
    run_scan(
        source,
        dest,
        _Fake({name: [hit] for name in ("a.jpg", "b.jpg", "c.jpg")}),
        [_person(tmp_path, False)],
        ScanOptions(),
        hooks=scan.ScanHooks(archive=lambda output: archive_report(output, data_root)),
    )
    assert seen_complete == [False, False, False, True]
    assert sorted(path.name for path in dest.iterdir()) == ["Emre", "rapor.json"]
    archived = list((data_root / "rapor-arsivi").rglob("*.json"))
    assert len(archived) == 1 and "2026-11-05T23-11-03" in archived[0].name
    assert '"results": []' in archived[0].read_text(encoding="utf-8")


def test_scan_copies_are_apfs_clones_of_the_resolved_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "in"
    source.mkdir()
    (source / "ikisi.jpg").write_bytes(b"photo")
    calls: list[list[str]] = []

    def cloned(argv, **_kwargs):
        calls.append(list(argv))
        Path(argv[-1]).write_bytes(Path(argv[-2]).read_bytes())
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(media, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(media.subprocess, "run", cloned)
    hit = PersonHit("emre", "Emre", 0.8, 1, False)
    dest = tmp_path / "out"
    run_scan(source, dest, _Fake({"ikisi.jpg": [hit]}), [_person(tmp_path, False)], ScanOptions())
    assert calls == [
        [
            "cp",
            "-c",
            "-p",
            str(source.resolve() / "ikisi.jpg"),
            str(dest.resolve() / "Emre" / "ikisi.jpg"),
        ]
    ]
    assert (dest / "Emre" / "ikisi.jpg").read_bytes() == b"photo"


def test_failed_clone_falls_back_to_a_plain_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "in"
    source.mkdir()
    (source / "ikisi.jpg").write_bytes(b"photo")
    calls: list[list[str]] = []

    def refused(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(media, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(media.subprocess, "run", refused)
    hit = PersonHit("emre", "Emre", 0.8, 1, False)
    dest = tmp_path / "out"
    run_scan(source, dest, _Fake({"ikisi.jpg": [hit]}), [_person(tmp_path, False)], ScanOptions())
    assert len(calls) == 1
    assert (dest / "Emre" / "ikisi.jpg").read_bytes() == b"photo"


def test_pool_source_keeps_camera_names(tmp_path: Path) -> None:
    pool = tmp_path / "havuz"
    (pool / "foto").mkdir(parents=True)
    (pool / "foto" / "_DSF4873.jpg").write_bytes(b"photo")
    (pool / "havuz.json").write_text("{}", encoding="utf-8")
    hit = PersonHit("emre", "Emre", 0.8, 1, False)
    dest = tmp_path / "out"
    run_scan(pool, dest, _Fake({"_DSF4873.jpg": [hit]}), [_person(tmp_path, False)], ScanOptions())
    assert [path.name for path in (dest / "Emre").iterdir()] == ["_DSF4873.jpg"]
