"""Coverage for the continuous webcam sweep. One angle never finishes it."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from recognize.camera_flow import CameraSession, save_enrollment
from recognize.gallery import ReferenceSample
from recognize.sweep import FaceSweep


def test_camera_frame_without_a_step_guide_is_not_a_server_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RECOGNIZE_HOME", str(tmp_path))
    from recognize.web import app

    with TestClient(app) as client:
        started = client.post("/api/kamera/baslat", json={"name": "Baran Başaran"})
        assert started.status_code == 200
        token = started.json()["token"]
        response = client.post(
            "/api/kamera/kare",
            data={"token": token},
            files={"frame": ("kare.jpg", b"not-a-jpeg", "image/jpeg")},
        )
    assert response.status_code == 400
    assert isinstance(response.json()["detail"], str)


def test_measured_baran_yaw_counts_the_users_right() -> None:
    """Yaw re-detected on Baran's saved crops. The access log prints no yaw.

    01.jpg düz +0.031, 03.jpg sağ +0.403 (delta +0.372), 02.jpg sol -0.329
    (delta -0.360). Positive is the user's right: 03.jpg is kamera-sag.jpg.
    05.jpg aşağı drifts to +0.114 (delta +0.083), which is outside the straight
    band, so that crop is a slight right. A chin-down whose yaw stays inside
    the band is still down.
    """
    from recognize.pose import REL_YAW_TURN, classify_from_baseline

    neutral_yaw = 0.0313
    neutral_pitch = 0.5918
    assert REL_YAW_TURN <= 0.09
    assert classify_from_baseline(0.0313, 0.5918, neutral_yaw, neutral_pitch) == "center"
    assert classify_from_baseline(0.4033, 0.5193, neutral_yaw, neutral_pitch) == "right"
    assert classify_from_baseline(-0.3292, 0.5594, neutral_yaw, neutral_pitch) == "left"
    assert classify_from_baseline(neutral_yaw + 0.02, 0.6539, neutral_yaw, neutral_pitch) == "down"
    assert classify_from_baseline(0.0872, 0.5553, neutral_yaw, neutral_pitch) == "up"

    sweep = FaceSweep()
    sweep.lock_neutral(neutral_yaw, neutral_pitch)
    right = sweep.note(0.4033, 0.5193, 0.0)
    assert right.kept is True
    assert sweep.count("right") == 1
    assert "Sağa bakış alındı." in right.message
    left = sweep.note(-0.3292, 0.5594, 1.0)
    assert left.kept is True
    assert "Sola bakış alındı." in left.message
    straight = sweep.note(neutral_yaw, neutral_pitch, 2.0)
    assert straight.kept is True
    assert sweep.records[-1].pose == "center"


def test_modest_right_turn_beats_a_pitch_drift() -> None:
    """Yaw +0.12 with pitch -0.05 is right. Yaw +0.02 with the same pitch is up."""
    from recognize.pose import classify_from_baseline

    baseline_yaw = 0.0
    baseline_pitch = 0.60
    right = classify_from_baseline(
        baseline_yaw + 0.12,
        baseline_pitch - 0.05,
        baseline_yaw,
        baseline_pitch,
    )
    look_up = classify_from_baseline(
        baseline_yaw + 0.02,
        baseline_pitch - 0.05,
        baseline_yaw,
        baseline_pitch,
    )
    assert right == "right"
    assert look_up == "up"


def test_measured_baran_pitches_count_as_up_and_down() -> None:
    """Pitches re-detected on Baran's saved crops.

    01.jpg düz 0.592, 04.jpg yukarı 0.555 (delta -0.037), 05.jpg aşağı 0.654
    (delta +0.062). The 0.06 gate calls the up frame unknown.
    Eye fraction on those aligned crops moved at most 0.020, so that is noise.
    """
    from recognize.pose import classify_from_baseline, landmarks_hold

    neutral = 0.592
    assert classify_from_baseline(0.0, neutral, 0.0, neutral) == "center"
    assert classify_from_baseline(0.0, 0.555, 0.0, neutral) == "up"
    assert classify_from_baseline(0.0, 0.654, 0.0, neutral) == "down"
    assert classify_from_baseline(0.0, neutral, 0.0, neutral, 0.47, 0.45) == "center"
    assert classify_from_baseline(0.0, neutral, 0.0, neutral, 0.50, 0.45) == "down"
    right_eye = (200.0, 100.0)
    left_eye = (100.0, 100.0)
    nose = (150.0, 130.0)
    mouth_right = (180.0, 145.0)
    mouth_left = (120.0, 145.0)
    assert landmarks_hold(right_eye, left_eye, nose, mouth_right, mouth_left) is False


def _with_neutral(yaw: float = 0.0, pitch: float = 0.58) -> FaceSweep:
    sweep = FaceSweep()
    sweep.lock_neutral(yaw, pitch)
    return sweep


def test_one_frame_duplicates_and_left_only_do_not_finish() -> None:
    sweep = _with_neutral()
    first = sweep.note(0.0, 0.58, 0.0)
    assert first.kept is True
    assert first.covered is False

    duplicate = sweep.note(0.01, 0.58, 0.05)
    assert duplicate.kept is False
    assert sweep.count("center") == 1

    turned = sweep.note(-0.30, 0.58, 0.08)
    assert turned.kept is True
    assert sweep.count("left") == 1

    for index, yaw in enumerate((-0.40, -0.55, -0.70)):
        assert sweep.note(yaw, 0.58, 1.0 + index * 0.2).kept is True
    extra = sweep.note(-0.85, 0.58, 2.0)
    assert extra.kept is False
    assert sweep.count("left") == 4
    assert sweep.covered is False


def test_surplus_turns_do_not_block_the_missing_down() -> None:
    """This session stored 40 samples and aşağı stayed at 2/3.

    Left 11, right 9, up 13 and center 5 had already passed their quotas.
    """
    sweep = _with_neutral(0.111, 0.60)
    moment = 0.0

    def add(yaw: float, pitch: float):
        nonlocal moment
        moment += 0.2
        return sweep.note(yaw, pitch, moment)

    for index in range(15):
        add(0.111 - 0.25 - index * 0.05, 0.60)
    assert sweep.count("left") == 4
    for index in range(15):
        add(0.111 + 0.25 + index * 0.05, 0.60)
    assert sweep.count("right") == 4
    assert add(0.104, 0.674).kept is True
    assert add(0.074, 0.726).kept is True
    assert add(0.056, 0.806).kept is True
    assert sweep.count("down") == 3
    assert len(sweep.records) < 40


def test_full_path_covers_every_direction() -> None:
    sweep = _with_neutral()
    moment = 0.0

    def add(yaw: float, pitch: float) -> None:
        nonlocal moment
        tick = sweep.note(yaw, pitch, moment)
        assert tick.kept is True
        moment += 0.2

    add(0.0, 0.58)
    add(0.0, 0.60)
    for yaw in (-0.25, -0.40, -0.55, -0.70):
        add(yaw, 0.58)
    for yaw in (0.25, 0.40, 0.55, 0.70):
        add(yaw, 0.58)
    for pitch in (0.50, 0.45, 0.40):
        add(0.0, pitch)
    for pitch in (0.72, 0.78, 0.86):
        add(0.0, pitch)

    assert sweep.count("center") >= 2
    assert sweep.count("left") >= 4
    assert sweep.count("right") >= 4
    assert sweep.count("up") >= 3
    assert sweep.count("down") >= 3
    assert sweep.covered is True


def test_saved_sweep_keeps_every_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECOGNIZE_HOME", str(tmp_path))
    sweep = _with_neutral()
    moment = 0.0
    views = [(0.0, 0.58), (0.0, 0.60)]
    views += [(-0.25 - 0.15 * index, 0.58) for index in range(4)]
    views += [(0.25 + 0.15 * index, 0.58) for index in range(4)]
    views += [(0.0, pitch) for pitch in (0.50, 0.45, 0.40, 0.72, 0.78, 0.86)]
    for yaw, pitch in views:
        assert sweep.note(yaw, pitch, moment).kept is True
        moment += 0.2
    assert sweep.covered is True

    session = CameraSession(name="Baran Başaran", samples=[], sweep=sweep)
    for index, record in enumerate(sweep.records):
        vector = np.zeros(4, dtype=np.float32)
        vector[index % 4] = 1.0
        session.samples.append(
            ReferenceSample(
                vector,
                0.9,
                b"jpeg",
                f"kamera-{index:02d}.jpg",
                record.yaw,
                record.pitch,
                tuple(float(point) for point in range(10)),
            )
        )
    person, warning = save_enrollment(session, root=tmp_path)
    assert warning is None
    assert person.clothing is None
    assert person.embeddings.shape == (len(sweep.records), 4)
    assert not np.allclose(person.embeddings[0], person.embeddings.mean(axis=0))
    payload = json.loads((tmp_path / person.slug / "person.json").read_text(encoding="utf-8"))
    assert payload["name"] == "Baran Başaran"
    assert payload["clothing"] is None
    assert payload["references"][0]["landmarks"] == [float(point) for point in range(10)]
    assert payload["references"][0]["yaw"] == pytest.approx(sweep.records[0].yaw)
    assert payload["references"][0]["pitch"] == pytest.approx(sweep.records[0].pitch)


def _yunet_face(nose_x: float, nose_y: float):
    from recognize.engine import DetectedFace

    # Mirrored frame. Box, right eye, left eye, nose, right mouth, left mouth.
    row = np.asarray(
        [40, 30, 160, 200, 200, 100, 100, 100, nose_x, nose_y, 180, 200, 120, 200],
        dtype=np.float32,
    )
    return DetectedFace(row=row, score=0.92, x=40, y=30, width=160, height=200)


class _OneFace:
    def __init__(self) -> None:
        self.face = _yunet_face(150.0, 158.0)

    def detect(self, image: np.ndarray, score_threshold: float):
        del image, score_threshold
        return [self.face]

    def embed(self, image: np.ndarray, face) -> np.ndarray:
        del image, face
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def crop_jpeg(self, image: np.ndarray, face, quality: int = 85) -> bytes:
        del image, face, quality
        return b"\xff\xd8\xff\xd9"


def _lit_frame(mean: int = 128) -> np.ndarray:
    rng = np.random.default_rng(7)
    noise = rng.integers(-40, 40, size=(320, 320, 3))
    return np.clip(noise + mean, 0, 255).astype(np.uint8)


def test_sweep_frame_reports_box_gaze_and_names_the_kept_direction() -> None:
    from recognize.camera_flow import record_sweep_frame

    engine = _OneFace()
    session = CameraSession(name="Baran Başaran", samples=[], sweep=FaceSweep())
    frame = _lit_frame()
    reports = [record_sweep_frame(engine, session, frame, 0.2 * index) for index in range(4)]
    assert [report.hint for report in reports[:3]] == ["learning"] * 3
    assert reports[0].gaze is None
    assert reports[0].box == pytest.approx((40 / 320, 30 / 320, 160 / 320, 200 / 320), abs=1e-3)
    assert session.sweep.neutral_pitch is not None

    engine.face = _yunet_face(120.0, 158.0)
    turned = record_sweep_frame(engine, session, frame, 2.0)
    assert turned.captured is True
    assert turned.hint == "captured"
    assert turned.pose == "left"
    assert turned.gaze is not None and turned.gaze[0] < -0.5
    assert session.samples[-1].source_name.startswith("kamera-left-")
    assert all("unknown" not in sample.source_name for sample in session.samples)


def test_dark_face_is_refused_with_a_hint() -> None:
    from recognize.camera_flow import record_sweep_frame

    session = CameraSession(name="Baran Başaran", samples=[], sweep=FaceSweep())
    report = record_sweep_frame(_OneFace(), session, _lit_frame(mean=20), 0.0)
    assert report.hint == "dark"
    assert report.captured is False
    assert session.samples == []


def test_next_region_follows_the_head_then_the_first_gap() -> None:
    sweep = _with_neutral()
    assert sweep.headline == "Başını yavaşça sola çevir"
    sweep.note(0.30, 0.58, 0.0)
    assert sweep.next_region == "right"
    assert sweep.headline == "Başını yavaşça sağa çevir"
    right = sweep.gaze(0.30, 0.58)
    up = sweep.gaze(0.0, 0.52)
    assert right is not None and right[0] > 0.9
    assert up is not None and up[1] > 0.9
    assert sweep.pose_of(-0.30, 0.58) == "left"
    assert FaceSweep().pose_of(-0.30, 0.58) == "unknown"


def test_camera_start_refuses_when_the_person_has_no_room(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from recognize.gallery import MAX_REFERENCES, add_references
    from recognize.sweep import SWEEP_TOTAL

    monkeypatch.setenv("RECOGNIZE_HOME", str(tmp_path))
    rows = [
        (np.eye(4, dtype=np.float32)[index % 4], 0.9, b"jpeg", f"{index}.jpg")
        for index in range(MAX_REFERENCES - SWEEP_TOTAL + 1)
    ]
    add_references("Baran Başaran", rows, None, update_clothing=False)
    from recognize.web import app

    with TestClient(app) as client:
        response = client.post("/api/kamera/baslat", json={"name": "Baran Başaran"})
    assert response.status_code == 400
    assert "yer yok" in response.json()["detail"]


def test_scan_rejects_an_empty_or_stale_person_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from recognize.gallery import add_references

    monkeypatch.setenv("RECOGNIZE_HOME", str(tmp_path))
    add_references(
        "Baran Başaran",
        [(np.eye(4, dtype=np.float32)[0], 0.9, b"jpeg", "a.jpg")],
        None,
        update_clothing=False,
    )
    source = tmp_path / "etkinlik"
    source.mkdir()
    from recognize.web import app

    body = {"source": str(source), "destination": str(tmp_path / "cikti")}
    with TestClient(app) as client:
        empty = client.post("/api/tara", json={**body, "people": []})
        stale = client.post("/api/tara", json={**body, "people": ["silinmis-kisi"]})
    assert empty.status_code == 400
    assert "en az bir kişi" in empty.json()["detail"]
    assert stale.status_code == 400


def test_a_small_or_unsure_second_detection_does_not_block_the_sweep() -> None:
    from recognize.camera_flow import record_sweep_frame
    from recognize.engine import DetectedFace

    blob = DetectedFace(row=np.zeros(14, np.float32), score=0.55, x=5, y=5, width=30, height=30)
    poster = DetectedFace(row=np.zeros(14, np.float32), score=0.9, x=250, y=5, width=50, height=60)
    friend = DetectedFace(
        row=np.zeros(14, np.float32), score=0.9, x=200, y=30, width=120, height=150
    )

    class _WithExtra(_OneFace):
        def __init__(self, extra) -> None:
            super().__init__()
            self.extra = extra

        def detect(self, image: np.ndarray, score_threshold: float):
            del image, score_threshold
            return [self.extra, self.face]

    frame = _lit_frame()
    for extra in (blob, poster):
        session = CameraSession(name="Baran Başaran", samples=[], sweep=FaceSweep())
        report = record_sweep_frame(_WithExtra(extra), session, frame, 0.0)
        assert report.hint == "learning"
    session = CameraSession(name="Baran Başaran", samples=[], sweep=FaceSweep())
    report = record_sweep_frame(_WithExtra(friend), session, frame, 0.0)
    assert report.hint == "many_faces"
