from __future__ import annotations

from pathlib import Path

from transcriber.files import collect_files


def test_collects_audio_recursively_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "call10.mp3").write_bytes(b"x")
    (tmp_path / "Call1.WAV").write_bytes(b"x")
    (tmp_path / "call2.mp3").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("не аудио", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "deep3.mp3").write_bytes(b"x")

    files, errors = collect_files([str(tmp_path)], recursive=True)
    names = [path.name for path in files]
    assert errors == []
    assert names == ["Call1.WAV", "call2.mp3", "call10.mp3", "deep3.mp3"]


def test_non_recursive_skips_nested(tmp_path: Path) -> None:
    (tmp_path / "a.mp3").write_bytes(b"x")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.mp3").write_bytes(b"x")

    files, errors = collect_files([str(tmp_path)], recursive=False)
    assert errors == []
    assert [path.name for path in files] == ["a.mp3"]


def test_explicit_file_is_accepted_with_any_extension(tmp_path: Path) -> None:
    weird = tmp_path / "recording.bin"
    weird.write_bytes(b"x")

    files, errors = collect_files([str(weird)])
    assert errors == []
    assert files == [weird]


def test_missing_path_reports_error(tmp_path: Path) -> None:
    files, errors = collect_files([str(tmp_path / "nope.mp3")])
    assert files == []
    assert errors and "не найден" in errors[0]
