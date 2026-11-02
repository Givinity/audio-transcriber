from __future__ import annotations

import io

import pytest

from transcriber.cli import parse_formats


def test_parse_formats_accepts_commas_and_spaces() -> None:
    assert parse_formats("txt, srt") == ["txt", "srt"]
    assert parse_formats("TXT;json") == ["txt", "json"]
    assert parse_formats(".txt") == ["txt"]


def test_parse_formats_deduplicates() -> None:
    assert parse_formats("txt,txt,srt") == ["txt", "srt"]


def test_parse_formats_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="неизвестный формат"):
        parse_formats("docx")


def test_parse_formats_rejects_empty() -> None:
    with pytest.raises(ValueError, match="ни один формат"):
        parse_formats(" , ")


def test_cuda_runtime_error_detection() -> None:
    from transcriber.models import is_cuda_runtime_error

    assert is_cuda_runtime_error(
        "Library libcublas.so.12 is not found or cannot be loaded"
    )
    assert is_cuda_runtime_error("Library libcudnn.so.9 is not found")
    assert is_cuda_runtime_error("CUDA failed with error out of memory")
    assert not is_cuda_runtime_error("ffmpeg не смог декодировать файл")
    assert not is_cuda_runtime_error("речь не распознана")


def test_cuda_failure_falls_back_to_cpu(monkeypatch, tmp_path) -> None:
    from transcriber import cli, pipeline
    from transcriber.models import LoadedModel

    cuda_model = LoadedModel(object(), "small", "cuda", "float16")
    cpu_model = LoadedModel(object(), "small", "cpu", "int8")
    calls: list[str] = []

    def fake_transcribe(path, model, options, progress=None):
        calls.append(model.device)
        if model.device == "cuda":
            raise pipeline.TranscriptionError(
                "Library libcublas.so.12 is not found or cannot be loaded"
            )
        return "готовый текст"

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(cli, "load_model", lambda **kwargs: cpu_model)

    warnings: list[str] = []
    reporter = cli.Reporter(stream=io.StringIO())
    monkeypatch.setattr(reporter, "warn", warnings.append)

    result, model = cli._transcribe_with_fallback(
        tmp_path / "call.mp3",
        cuda_model,
        pipeline.Options(),
        cli.Progress(enabled=False),
        reporter,
    )

    assert result == "готовый текст"
    assert model is cpu_model
    assert calls == ["cuda", "cpu"]
    assert warnings and "libcublas" in warnings[0]
