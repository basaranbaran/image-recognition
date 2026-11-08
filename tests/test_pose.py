"""Pose bands and the rule that a wrong pose is not enrolled."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from recognize.camera_flow import CameraSession, assess_frame, save_enrollment
from recognize.engine import DetectedFace
from recognize.errors import AppError
from recognize.pose import (
    CENTER_PITCH_MAX,
    CENTER_PITCH_MIN,
    CENTER_YAW_MAX,
    DOWN_PITCH_MIN,
    HOLD_FRAMES,
    MIN_DET_SCORE,
    MIN_FACE_PX,
    STEPS,
    TURN_YAW_MIN,
    UP_PITCH_MAX,
    PoseGuide,
    classify_pose,
    landmarks_hold,
    pose_from_yunet,
    sequence_line,
    yaw_pitch,
)

REAL_PEOPLE = Path(__file__).parents[1] / "veri" / "kisiler"


def _tree(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {str(path.relative_to(root)) for path in root.rglob("*")}


@pytest.fixture(autouse=True)
def gallery_untouched() -> None:
    before = _tree(REAL_PEOPLE)
    yield
    assert _tree(REAL_PEOPLE) == before


def _row(nose_x: float, nose_y: float) -> np.ndarray:
    # Mirrored frame: image-left is the user's left.
    # YuNet order: box, right eye, left eye, nose, right mouth, left mouth, score.
    values = [
        40,
        30,
        160,
        200,
        200,
        100,
        100,
        100,
        nose_x,
        nose_y,
        180,
        200,
        120,
        200,
        0.95,
    ]
    return np.asarray(values, dtype=np.float32)


def _face(kind: str, *, px: float = 180, score: float = 0.92) -> DetectedFace:
    noses = {
        "center": (150.0, 158.0),
        "left": (120.0, 158.0),
        "right": (180.0, 158.0),
        "up": (150.0, 150.0),
        "down": (150.0, 175.0),
    }
    nose_x, nose_y = noses[kind]
    return DetectedFace(
        row=_row(nose_x, nose_y)[:14],
        score=score,
        x=10,
        y=10,
        width=px,
        height=px + 30,
    )


def _image() -> np.ndarray:
    return np.zeros((32, 32, 3), dtype=np.uint8)


class _Engine:
    def __init__(self, face: DetectedFace) -> None:
        self.face = face
        self.embeds = 0

    def detect(self, image: np.ndarray, score_threshold: float) -> list[DetectedFace]:
        del image, score_threshold
        return [self.face]

    def embed(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        del image, face
        self.embeds += 1
        return np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def crop_jpeg(self, image: np.ndarray, face: DetectedFace, quality: int = 85) -> bytes:
        del image, face, quality
        return b"\xff\xd8\xff\xd9"


class _Flaky(_Engine):
    def embed(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        del image, face
        raise ValueError("no feature")


class _Crowd:
    def __init__(self) -> None:
        self.face = _face("left")

    def detect(self, image: np.ndarray, score_threshold: float) -> list[DetectedFace]:
        del image, score_threshold
        return [self.face, self.face]

    def embed(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        del image, face
        raise AssertionError("two faces must not be embedded")

    def crop_jpeg(self, image: np.ndarray, face: DetectedFace, quality: int = 85) -> bytes:
        del image, face, quality
        raise AssertionError("two faces must not be cropped")


class _Empty:
    def detect(self, image: np.ndarray, score_threshold: float) -> list[DetectedFace]:
        del image, score_threshold
        return []

    def embed(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        del image, face
        raise AssertionError("missing face must not be embedded")

    def crop_jpeg(self, image: np.ndarray, face: DetectedFace, quality: int = 85) -> bytes:
        del image, face, quality
        raise AssertionError("missing face must not be cropped")


def _session() -> CameraSession:
    return CameraSession(name="Baran Başaran", guide=PoseGuide(), samples=[])


def _reach(step_id: str) -> CameraSession:
    session = _session()
    image = _image()
    for step in STEPS:
        if step.id == step_id:
            assert session.guide.current is not None
            assert session.guide.current.id == step_id
            return session
        assert step.required
        engine = _Engine(_face(step.id))
        captured = False
        for _ in range(HOLD_FRAMES):
            report = assess_frame(engine, session, image)
            captured = report.captured or captured
        assert captured
        assert engine.embeds == 1
    raise AssertionError(step_id)


def test_user_left_on_a_mirrored_frame_is_negative_yaw() -> None:
    cases = (
        (150.0, 158.0, "center", 0.0, 0.58),
        (120.0, 158.0, "left", -0.3, 0.58),
        (180.0, 158.0, "right", 0.3, 0.58),
        (150.0, 150.0, "up", 0.0, 0.50),
        (150.0, 175.0, "down", 0.0, 0.75),
    )
    for nose_x, nose_y, name, yaw, pitch in cases:
        full = _row(nose_x, nose_y)
        got_yaw, got_pitch, got_name = pose_from_yunet(full)
        assert got_name == name
        assert got_yaw == pytest.approx(yaw)
        assert got_pitch == pytest.approx(pitch)
        short_yaw, short_pitch, short_name = pose_from_yunet(full[:14])
        assert short_name == name
        assert short_yaw == pytest.approx(got_yaw)
        assert short_pitch == pytest.approx(got_pitch)
    assert pose_from_yunet(_row(120.0, 150.0))[0] < 0
    assert pose_from_yunet(_row(180.0, 150.0))[0] > 0


def test_short_yunet_row_is_rejected() -> None:
    with pytest.raises(ValueError):
        pose_from_yunet(np.zeros(13, dtype=np.float32))


def test_pose_bands_leave_a_gap_around_frontal() -> None:
    assert TURN_YAW_MIN > CENTER_YAW_MAX
    assert UP_PITCH_MAX < CENTER_PITCH_MIN
    assert DOWN_PITCH_MIN > CENTER_PITCH_MAX
    gap = (TURN_YAW_MIN + CENTER_YAW_MAX) / 2
    assert classify_pose(gap, 0.58) == "unknown"
    assert classify_pose(-gap, 0.58) == "unknown"
    assert classify_pose(0.0, 0.58) == "center"
    assert classify_pose(CENTER_YAW_MAX, 0.58) == "center"
    assert classify_pose(CENTER_YAW_MAX + 0.011, 0.58) == "unknown"
    assert classify_pose(-(TURN_YAW_MIN - 0.011), 0.58) == "unknown"
    assert classify_pose(-TURN_YAW_MIN, 0.58) == "left"
    assert classify_pose(TURN_YAW_MIN, 0.58) == "right"
    assert classify_pose(0.0, UP_PITCH_MAX) == "up"
    assert classify_pose(0.0, UP_PITCH_MAX + 0.011) == "unknown"
    assert classify_pose(0.0, CENTER_PITCH_MIN) == "center"
    assert classify_pose(0.0, CENTER_PITCH_MAX) == "center"
    assert classify_pose(0.0, DOWN_PITCH_MIN) == "down"
    assert classify_pose(0.0, DOWN_PITCH_MIN - 0.011) == "unknown"


def test_slight_look_up_passes_and_straight_does_not() -> None:
    """Pitch 0.50 is a slight look-up. The old ceiling (0.30) called it straight.

    Eyes at y=100 and mouth at y=200 put the nose at y=150 for this pitch.
    A straight face, nose at y=158, stays center.
    """
    slight = _row(150.0, 150.0)
    straight = _row(150.0, 158.0)
    assert pose_from_yunet(slight)[2] == "up"
    assert pose_from_yunet(slight)[1] == pytest.approx(0.50)
    assert pose_from_yunet(straight)[2] == "center"
    assert pose_from_yunet(straight)[1] == pytest.approx(0.58)
    assert classify_pose(0.0, 0.50) == "up"
    assert classify_pose(0.0, 0.30) == "unknown"
    assert classify_pose(0.0, 0.58) == "center"


def test_natural_chin_down_passes_and_crushed_pitch_does_not() -> None:
    """0.66 is a slight chin-down. The old floor (0.70) left it unknown.

    0.20 puts the nose on the eyes. That used to count as up and saved a bad crop.
    """
    assert classify_pose(0.0, 0.66) == "down"
    assert classify_pose(0.0, 0.58) == "center"
    assert classify_pose(0.0, 0.50) == "up"
    assert classify_pose(0.0, 0.20) == "unknown"
    assert classify_pose(0.0, 1.05) == "unknown"
    chin_down = _row(150.0, 166.0)
    assert pose_from_yunet(chin_down)[1] == pytest.approx(0.66)
    assert pose_from_yunet(chin_down)[2] == "down"


def test_collapsed_landmarks_are_not_a_usable_capture() -> None:
    # Eyes 100px apart, mouth only 45px below them. Pitch still looks like down.
    right_eye = (200.0, 100.0)
    left_eye = (100.0, 100.0)
    nose = (150.0, 130.0)
    mouth_right = (180.0, 145.0)
    mouth_left = (120.0, 145.0)
    _yaw, pitch = yaw_pitch(right_eye, left_eye, nose, mouth_right, mouth_left)
    assert classify_pose(0.0, pitch) == "down"
    assert landmarks_hold(right_eye, left_eye, nose, mouth_right, mouth_left) is False


def test_small_or_dim_face_does_not_count() -> None:
    guide = PoseGuide()
    small = guide.observe("center", MIN_FACE_PX - 1, 0.95)
    assert small.ready is False
    assert "yaklaşın" in small.message
    dim = guide.observe("center", float(MIN_FACE_PX), MIN_DET_SCORE - 0.01)
    assert dim.ready is False
    assert "net değil" in dim.message
    held = guide.observe("center", float(MIN_FACE_PX), MIN_DET_SCORE)
    assert held.matching is True
    assert held.streak == 1
    with pytest.raises(AppError):
        guide.commit()


@pytest.mark.parametrize("kind", ["center", "right", "up", "down"])
def test_left_step_rejects_other_poses(kind: str) -> None:
    session = _reach("left")
    engine = _Engine(_face(kind))
    report = None
    for _ in range(HOLD_FRAMES + 2):
        report = assess_frame(engine, session, _image())
        assert report.captured is False
        assert report.matches is False
    assert report is not None
    assert "kaydedilmedi" in report.message
    assert "Sola dön" in report.message
    if kind == "center":
        assert "Düz bakıyorsunuz" in report.message
    assert engine.embeds == 0
    assert [item[3] for item in session.samples] == ["kamera-duz.jpg"]
    assert session.guide.current is not None
    assert session.guide.current.id == "left"


def test_missing_face_resets_a_partial_hold() -> None:
    session = _reach("left")
    image = _image()
    left = _Engine(_face("left"))
    assess_frame(left, session, image)
    assess_frame(left, session, image)
    assert session.guide.streak == 2
    report = assess_frame(_Empty(), session, image)
    assert report.captured is False
    assert session.guide.streak == 0
    assert "görünmüyor" in report.message
    assert left.embeds == 0


def test_two_faces_are_not_captured() -> None:
    session = _reach("left")
    crowd = _Crowd()
    for _ in range(HOLD_FRAMES + 1):
        report = assess_frame(crowd, session, _image())
        assert report.captured is False
    assert "tek yüz" in report.message
    assert len(session.samples) == 1


def test_failed_embedding_is_not_stored() -> None:
    session = _reach("left")
    engine = _Flaky(_face("left"))
    for _ in range(HOLD_FRAMES):
        report = assess_frame(engine, session, _image())
        assert report.captured is False
    assert len(session.samples) == 1
    assert session.guide.current is not None
    assert session.guide.current.id == "left"


def test_closed_eyes_waits_for_the_button_then_holds() -> None:
    session = _reach("eyes")
    engine = _Engine(_face("center"))
    image = _image()
    for _ in range(HOLD_FRAMES + 3):
        report = assess_frame(engine, session, image)
        assert report.captured is False
    assert engine.embeds == 0
    session.guide.arm_eyes()
    captured_at = None
    for index in range(HOLD_FRAMES):
        report = assess_frame(engine, session, image)
        if report.captured:
            captured_at = index
    assert captured_at == HOLD_FRAMES - 1
    assert engine.embeds == 1
    assert session.samples[-1][3] == "kamera-goz.jpg"


def test_save_waits_for_required_poses_and_skips_clothing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RECOGNIZE_HOME", str(tmp_path))
    early = _reach("left")
    with pytest.raises(AppError, match="Zorunlu"):
        save_enrollment(early, root=tmp_path)
    assert not (tmp_path / "baran-basaran").exists()

    session = _reach("eyes")
    assert session.guide.required_done
    assert [item[3] for item in session.samples] == [
        "kamera-duz.jpg",
        "kamera-sol.jpg",
        "kamera-sag.jpg",
        "kamera-yukari.jpg",
        "kamera-asagi.jpg",
    ]
    person, warning = save_enrollment(session, root=tmp_path)
    assert warning is None
    assert person.name == "Baran Başaran"
    assert person.slug == "baran-basaran"
    assert person.clothing is None
    assert len(person.references) == 5
    assert person.embeddings.shape == (5, 4)
    text = (tmp_path / "baran-basaran" / "person.json").read_text(encoding="utf-8")
    assert '"name": "Baran Başaran"' in text
    assert '"clothing": null' in text

    session.guide.arm_eyes()
    engine = _Engine(_face("center"))
    captured = False
    for _ in range(HOLD_FRAMES):
        captured = assess_frame(engine, session, _image()).captured or captured
    assert captured
    again, _warning = save_enrollment(session, root=tmp_path)
    assert len(again.references) == 6
    assert again.clothing is None
    assert again.embeddings.shape == (6, 4)


def test_skipped_eyes_are_not_required(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RECOGNIZE_HOME", str(tmp_path))
    session = _reach("eyes")
    session.guide.skip()
    assert session.guide.finished
    person, _warning = save_enrollment(session, root=tmp_path)
    assert len(person.references) == 5
    assert person.clothing is None
    payload = json.loads((tmp_path / person.slug / "person.json").read_text(encoding="utf-8"))
    assert payload["clothing"] is None
    assert payload["name"] == "Baran Başaran"


def test_sequence_lists_poses_in_order() -> None:
    text = sequence_line()
    assert text.index("Düz bak") < text.index("Sola dön") < text.index("Sağa dön")
    assert text.index("Sağa dön") < text.index("Biraz yukarı bak") < text.index("Biraz aşağı bak")
    assert "Gözlerini kapat (atlanabilir)" in text


def test_camera_page_does_not_write_a_person(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RECOGNIZE_HOME", str(tmp_path))
    from recognize.web import app

    with TestClient(app) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "Kamerayla tanıt" in home.text
        page = client.get("/kamera")
        assert page.status_code == 200
        assert "Baran Başaran" not in page.text
        assert "Düz bak" in page.text
        assert "Sola dön" in page.text
        unnamed = client.post("/api/kamera/baslat", json={})
        assert unnamed.status_code == 400
        assert unnamed.json()["detail"] == "Kişi adı boş olamaz."
        started = client.post("/api/kamera/baslat", json={"name": "Baran Başaran"})
        assert started.status_code == 200
        body = started.json()
        assert body["name"] == "Baran Başaran"
        assert body["instruction"] == "Önce düz bak, yüzünü öğreniyorum"
        assert body["ready"] is False
        assert [row["label"] for row in body["coverage"]] == [
            "Sola dön",
            "Sağa dön",
            "Biraz yukarı bak",
            "Biraz aşağı bak",
            "Düz bak",
        ]
        saved = client.post("/api/kamera/kaydet", json={"token": body["token"]})
        assert saved.status_code == 400
    assert not (tmp_path / "baran-basaran").exists()
    assert not (tmp_path / "veri" / "kisiler" / "baran-basaran").exists()


def test_camera_start_is_blocked_while_a_panel_scan_runs() -> None:
    from recognize import web

    web._scan["state"] = "running"
    try:
        with TestClient(web.app) as client:
            response = client.post("/api/kamera/baslat", json={"name": "Baran Başaran"})
            assert response.status_code == 409
    finally:
        web._scan["state"] = "idle"
