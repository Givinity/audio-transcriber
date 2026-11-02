from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .audio import decode_audio
from .models import LoadedModel
from .output import Segment

ProgressCallback = Callable[[float, float], None]


class TranscriptionError(RuntimeError):
    """Ошибка во время распознавания."""


@dataclass
class Options:
    model: str = "small"
    language: str | None = None
    device: str = "auto"
    compute_type: str | None = None
    beam_size: int = 5
    vad: bool = True
    denoise: str = "none"
    prompt: str | None = None
    word_timestamps: bool = False
    cache_dir: str | None = None
    cpu_threads: int = 0
    offline: bool = False


@dataclass
class Transcript:
    path: Path
    duration: float
    language: str
    language_probability: float
    segments: list[Segment]
    metadata: dict


def transcribe(
    path: Path,
    model: LoadedModel,
    options: Options,
    progress: ProgressCallback | None = None,
) -> Transcript:
    audio, duration = decode_audio(path, denoise=options.denoise)

    language = options.language
    if not language or language == "auto":
        language = None

    transcribe_options: dict = {
        "language": language,
        "task": "transcribe",
        "beam_size": options.beam_size,
        "vad_filter": options.vad,
        "condition_on_previous_text": False,
        "initial_prompt": options.prompt or None,
        "word_timestamps": options.word_timestamps,
        # Небольшой fallback по температуре уменьшает зацикливание на шуме.
        "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    }
    if options.vad:
        transcribe_options["vad_parameters"] = {
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        }

    try:
        segment_iterator, info = model.model.transcribe(audio, **transcribe_options)
    except Exception as exc:
        raise TranscriptionError(
            f"модель не смогла обработать «{path.name}»: {exc}"
        ) from exc

    segments: list[Segment] = []
    try:
        for raw in segment_iterator:
            text = (raw.text or "").strip()
            if not text:
                continue

            words = []
            if options.word_timestamps and getattr(raw, "words", None):
                words = [
                    {
                        "start": round(word.start, 3),
                        "end": round(word.end, 3),
                        "word": word.word,
                        "probability": round(word.probability, 4),
                    }
                    for word in raw.words
                ]

            segments.append(
                Segment(
                    start=float(raw.start),
                    end=float(raw.end),
                    text=text,
                    avg_logprob=float(raw.avg_logprob)
                    if raw.avg_logprob is not None
                    else None,
                    no_speech_prob=float(raw.no_speech_prob)
                    if raw.no_speech_prob is not None
                    else None,
                    words=words,
                )
            )
            if progress:
                progress(float(raw.end), duration)
    except Exception as exc:
        raise TranscriptionError(
            f"ошибка при распознавании «{path.name}»: {exc}"
        ) from exc

    if progress:
        progress(duration, duration)

    detected_language = getattr(info, "language", None) or language or "unknown"
    probability = getattr(info, "language_probability", 0.0) or 0.0

    metadata = {
        "file": str(path),
        "duration": round(duration, 3),
        "language": detected_language,
        "language_probability": round(float(probability), 4),
        "model": model.name,
        "device": model.device,
        "compute_type": model.compute_type,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {
            "task": "transcribe",
            "beam_size": options.beam_size,
            "vad": options.vad,
            "denoise": options.denoise,
            "word_timestamps": options.word_timestamps,
        },
    }

    return Transcript(
        path=path,
        duration=duration,
        language=detected_language,
        language_probability=float(probability),
        segments=segments,
        metadata=metadata,
    )
