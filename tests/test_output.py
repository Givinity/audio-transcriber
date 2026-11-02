from __future__ import annotations

from pathlib import Path

from transcriber.output import (
    Segment,
    format_duration,
    format_timestamp,
    plan_outputs,
    render_srt,
    render_txt,
    render_vtt,
    write_text_atomic,
)


def seg(start: float, end: float, text: str) -> Segment:
    return Segment(start=start, end=end, text=text)


def test_format_timestamp_srt_and_vtt() -> None:
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(3661.234) == "01:01:01,234"
    assert format_timestamp(3661.234, ".") == "01:01:01.234"
    assert format_timestamp(360000) == "100:00:00,000"


def test_format_duration() -> None:
    assert format_duration(0) == "0:00"
    assert format_duration(97) == "1:37"
    assert format_duration(3723) == "1:02:03"


def test_txt_paragraphs_use_pauses() -> None:
    segments = [
        seg(0, 2, "Привет."),
        seg(2.5, 4, "Как дела?"),
        seg(7, 8, "Всё хорошо."),
    ]
    assert render_txt(segments) == ("Привет. Как дела?\n\nВсё хорошо.\n")


def test_txt_with_timestamps() -> None:
    text = render_txt([seg(65.2, 66.0, "Проверка")], timestamps=True)
    assert text == "[00:01:05] Проверка\n"


def test_srt_blocks_are_numbered_and_wrapped() -> None:
    segments = [seg(0, 1.5, "Короткая фраза"), seg(2, 3, "Вторая фраза")]
    text = render_srt(segments)
    assert text.startswith("1\n00:00:00,000 --> 00:00:01,500\nКороткая фраза\n\n")
    assert "2\n00:00:02,000 --> 00:00:03,000\nВторая фраза\n" in text


def test_vtt_has_header_and_dot_separator() -> None:
    text = render_vtt([seg(1.25, 2.5, "Тест")])
    assert text.startswith("WEBVTT\n\n")
    assert "00:00:01.250 --> 00:00:02.500" in text


def test_plan_outputs_avoids_collisions(tmp_path: Path) -> None:
    (tmp_path / "call.txt").write_text("old", encoding="utf-8")
    reserved: set[Path] = set()
    plans = plan_outputs(
        input_path=Path("/input/call.mp3"),
        formats=["txt", "srt"],
        output_dir=tmp_path,
        reserved=reserved,
    )
    assert plans["txt"] == tmp_path / "call-2.txt"
    assert plans["srt"] == tmp_path / "call-2.srt"

    second = plan_outputs(
        input_path=Path("/input/call.mp3"),
        formats=["txt"],
        output_dir=tmp_path,
        reserved=reserved,
    )
    assert second["txt"] == tmp_path / "call-3.txt"


def test_write_text_atomic_leaves_no_tmp(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "result.txt"
    write_text_atomic(target, "текст\n")
    assert target.read_text(encoding="utf-8") == "текст\n"
    assert not (tmp_path / "nested" / "result.txt.tmp").exists()
