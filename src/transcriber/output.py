from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FORMATS = ("txt", "srt", "vtt", "json")


@dataclass
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    words: list[dict[str, Any]] = field(default_factory=list)


def format_timestamp(seconds: float, ms_separator: str = ",") -> str:
    """00:01:23,456 — часы не ограничены двумя цифрами."""
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{ms_separator}{millis:03d}"


def format_duration(seconds: float) -> str:
    """1:37 или 1:02:03 — для сообщений в консоли."""
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def wrap_text(text: str, width: int = 42) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def render_txt(
    segments: list[Segment],
    timestamps: bool = False,
    paragraph_gap: float = 1.2,
    max_chars: int = 1000,
) -> str:
    """Обычный текст: абзацы по паузам либо строки с таймкодами."""
    prepared = [(segment, " ".join(segment.text.split())) for segment in segments]
    prepared = [(segment, text) for segment, text in prepared if text]

    if timestamps:
        lines = [
            f"[{format_timestamp(segment.start)[:8]}] {text}"
            for segment, text in prepared
        ]
        return "\n".join(lines) + ("\n" if lines else "")

    paragraphs: list[str] = []
    current: list[str] = []
    current_length = 0
    previous_end: float | None = None

    for segment, text in prepared:
        gap = segment.start - previous_end if previous_end is not None else 0.0
        too_long = current and current_length + len(text) + 1 > max_chars
        if current and (gap > paragraph_gap or too_long):
            paragraphs.append(" ".join(current))
            current = []
            current_length = 0
        current.append(text)
        current_length += len(text) + 1
        previous_end = segment.end

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs) + ("\n" if paragraphs else "")


def render_srt(segments: list[Segment], width: int = 42) -> str:
    blocks: list[str] = []
    index = 0
    for segment in segments:
        text = " ".join(segment.text.split())
        if not text:
            continue
        index += 1
        lines = wrap_text(text, width) or [""]
        blocks.append(
            f"{index}\n"
            f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
            + "\n".join(lines)
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_vtt(segments: list[Segment], width: int = 42) -> str:
    blocks = ["WEBVTT"]
    for segment in segments:
        text = " ".join(segment.text.split())
        if not text:
            continue
        lines = wrap_text(text, width) or [""]
        blocks.append(
            f"{format_timestamp(segment.start, '.')} --> "
            f"{format_timestamp(segment.end, '.')}\n" + "\n".join(lines)
        )
    return "\n\n".join(blocks) + "\n"


def segment_to_dict(segment: Segment) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "start": round(segment.start, 3),
        "end": round(segment.end, 3),
        "text": segment.text,
    }
    if segment.avg_logprob is not None:
        payload["avg_logprob"] = round(segment.avg_logprob, 4)
    if segment.no_speech_prob is not None:
        payload["no_speech_prob"] = round(segment.no_speech_prob, 4)
    if segment.words:
        payload["words"] = segment.words
    return payload


def render_json(metadata: dict[str, Any], segments: list[Segment]) -> str:
    payload = dict(metadata)
    payload["segments"] = [segment_to_dict(segment) for segment in segments]
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def base_outputs(
    input_path: Path,
    formats: list[str],
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Базовые имена результатов без учёта коллизий."""
    directory = Path(output_dir).expanduser() if output_dir else input_path.parent
    return {fmt: directory / f"{input_path.stem}.{fmt}" for fmt in formats}


def plan_outputs(
    input_path: Path,
    formats: list[str],
    output_dir: str | Path | None,
    reserved: set[Path],
) -> dict[str, Path]:
    """Куда писать результаты.

    Без --output-dir файлы кладутся рядом с исходником и перезаписываются.
    С --output-dir одинаковые имена не затирают друг друга: появляется суффикс -2.
    """
    directory = Path(output_dir).expanduser() if output_dir else input_path.parent
    stem = input_path.stem

    if output_dir:
        candidate = stem
        counter = 2
        while True:
            paths = {fmt: directory / f"{candidate}.{fmt}" for fmt in formats}
            taken = any(path.exists() for path in paths.values()) or any(
                path in reserved for path in paths.values()
            )
            if not taken:
                stem = candidate
                break
            candidate = f"{stem}-{counter}"
            counter += 1

    paths = {fmt: directory / f"{stem}.{fmt}" for fmt in formats}
    reserved.update(paths.values())
    return paths
