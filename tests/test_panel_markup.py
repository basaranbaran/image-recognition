"""The panel shell: the theme is set before the first paint, and deleting a person asks first."""

from pathlib import Path

STATIC = Path(__file__).parents[1] / "src" / "recognize" / "static"


def test_theme_is_chosen_before_the_stylesheet_and_the_toggle_exists() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert html.index("/static/theme.js") < html.index("/static/app.css")
    assert 'id="theme-toggle"' in html
    assert ':root[data-theme="dark"]' in css


def test_deleting_a_person_asks_first_and_refreshes_the_archive() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'id="delete-dialog"' in html and 'id="delete-ok"' in html
    assert "confirmDeletePerson(person, card, remove)" in script
    assert script.count("await refreshResults();") == 2
