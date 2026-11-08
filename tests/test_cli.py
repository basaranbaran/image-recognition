"""Panel copy and CLI help stay in Turkish and mention the real constraints."""

from pathlib import Path

from typer.testing import CliRunner

from recognize.cli import app
from recognize.messages import ANGLE_GUIDANCE, CLOTHING_NOTE, ONE_PHOTO_NOTE, THRESHOLD_HINT

runner = CliRunner()


def test_panel_copy_mentions_angles_and_optional_yellow() -> None:
    html = (Path(__file__).parents[1] / "src" / "recognize" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    assert ANGLE_GUIDANCE in html
    assert ONE_PHOTO_NOTE in html
    assert CLOTHING_NOTE in html
    assert "kimsenin giymediği tek bir renk" in html
    assert 'value="sari"' in html
    assert THRESHOLD_HINT in html
    assert "0,5" in html


def test_enroll_help_asks_for_angles_and_clothing() -> None:
    result = runner.invoke(app, ["tanit", "--help"])
    assert result.exit_code == 0
    text = result.stdout
    assert "gözleri kapalı" in text
    assert "soldan" in text
    assert "sağdan" in text
    assert "--kiyafet" in text
    assert "sari" in text


def test_scan_help_keeps_the_stricter_threshold() -> None:
    result = runner.invoke(app, ["tara", "--help"])
    assert result.exit_code == 0
    assert "0.36" in result.stdout
    assert "0.5" in result.stdout
    assert "0.30" not in result.stdout


def test_camera_page_names_the_poses_but_no_person() -> None:
    root = Path(__file__).parents[1] / "src" / "recognize" / "static"
    page = (root / "kamera.html").read_text(encoding="utf-8")
    script = (root / "kamera.js").read_text(encoding="utf-8")
    index = (root / "index.html").read_text(encoding="utf-8")
    assert 'href="/kamera"' in index
    assert "Kamerayla tanıt" in index
    for phrase in (
        "Düz bak",
        "Sola dön",
        "Sağa dön",
        "Biraz yukarı bak",
        "Biraz aşağı bak",
        "derinlik sensörü yok",
    ):
        assert phrase in page
    assert "Baran Başaran" not in page
    folded = page.casefold()
    assert "kıyafet" not in folded
    assert "sari" not in folded
    assert "scale(-1" in script
    assert "getUserMedia" in script
    assert "alert(" not in script
    assert "http://" not in script


def test_camera_help_does_not_ask_for_clothing() -> None:
    result = runner.invoke(app, ["kamera", "--help"])
    assert result.exit_code == 0
    assert "--kiyafet" not in result.stdout
    assert "Tarayıcı" in result.stdout or "tarayıcı" in result.stdout


def test_camera_command_needs_a_name(monkeypatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    result = runner.invoke(app, ["kamera"])
    assert result.exit_code != 0
    assert opened == []


def test_camera_command_opens_the_page_without_enrolling(monkeypatch) -> None:
    opened: list[str] = []

    def _open(url: str) -> None:
        opened.append(url)

    def _boom(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("enrollment must wait for the camera")

    monkeypatch.setattr("webbrowser.open", _open)
    monkeypatch.setattr("recognize.cli.enroll_faces", _boom)
    people = Path(__file__).parents[1] / "veri" / "kisiler"
    before = {str(path.relative_to(people)) for path in people.rglob("*")}
    result = runner.invoke(app, ["kamera", "Baran Başaran"])
    assert result.exit_code == 0, result.stdout
    assert "127.0.0.1:8765/kamera" in result.stdout
    assert "Tek kare yetmez." in result.stdout
    assert "derinlik sensörü yok" in result.stdout
    assert "Kıyafet rengi eklenmez." in result.stdout
    assert opened
    assert "ad=Baran+Ba%C5%9Faran" in opened[0]
    after = {str(path.relative_to(people)) for path in people.rglob("*")}
    assert after == before


def test_camera_command_rejects_a_blank_name(monkeypatch) -> None:
    monkeypatch.setattr(
        "webbrowser.open",
        lambda url: (_ for _ in ()).throw(AssertionError(url)),
    )
    result = runner.invoke(app, ["kamera", "   "])
    assert result.exit_code == 1
    assert "boş" in result.stdout
