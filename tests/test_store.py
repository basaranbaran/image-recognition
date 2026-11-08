"""Atomic JSON writes and the recent-scan list the panel opens on."""

import json
from pathlib import Path

import pytest

from recognize.store import RECENT_LIMIT, read_json, recent_scans, remember_scan, write_json_atomic


def test_atomic_write_leaves_no_temporary_file_and_keeps_the_old_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "rapor.json"
    write_json_atomic(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rapor.json"]

    def boom(*_args, **_kwargs):
        raise ValueError("no")

    monkeypatch.setattr("recognize.store.json.dumps", boom)
    with pytest.raises(ValueError):
        write_json_atomic(target, {"a": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rapor.json"]


def test_recent_scans_put_the_newest_first_and_flag_missing_reports(tmp_path: Path) -> None:
    first = tmp_path / "cikti-1"
    second = tmp_path / "cikti-2"
    for folder in (first, second):
        folder.mkdir()
        (folder / "rapor.json").write_text("{}", encoding="utf-8")
    at = "2026-11-06T0{}:00:00+00:00"
    remember_scan(first, {"files": 3, "matchedFiles": 1}, at.format(1), root=tmp_path)
    remember_scan(second, {"files": 5, "matchedFiles": 2}, at.format(2), root=tmp_path)
    rows = recent_scans(tmp_path)
    assert [row["output"] for row in rows] == [str(second.resolve()), str(first.resolve())]
    assert rows[0]["matchedFiles"] == 2 and rows[0]["exists"] is True

    # Scanning the first folder again moves it to the head instead of adding a twin.
    remember_scan(first, {"files": 4, "matchedFiles": 4}, at.format(3), root=tmp_path)
    rows = recent_scans(tmp_path)
    assert [row["output"] for row in rows] == [str(first.resolve()), str(second.resolve())]
    assert rows[0]["files"] == 4

    (second / "rapor.json").unlink()
    assert recent_scans(tmp_path)[1]["exists"] is False


def test_recent_list_is_capped_and_survives_a_broken_file(tmp_path: Path) -> None:
    for index in range(RECENT_LIMIT + 3):
        folder = tmp_path / f"cikti-{index}"
        folder.mkdir()
        remember_scan(folder, None, f"2026-11-06T00:{index:02d}:00+00:00", root=tmp_path)
    assert len(recent_scans(tmp_path)) == RECENT_LIMIT
    (tmp_path / "son-taramalar.json").write_text("{not json", encoding="utf-8")
    assert recent_scans(tmp_path) == []
    assert read_json(tmp_path / "son-taramalar.json") is None
