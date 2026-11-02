from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "small"

MODEL_HINTS = {
    "tiny": "~75 МБ, самый быстрый, низкая точность",
    "base": "~145 МБ, быстрый, базовая точность",
    "small": "~484 МБ, разумный баланс для CPU",
    "medium": "~1.5 ГБ, заметно точнее, медленнее",
    "large-v3": "~3.1 ГБ, максимальная точность, нужен мощный CPU/GPU",
    "distil-large-v3": "~1.5 ГБ, быстрый вариант large-v3 (в основном английский)",
}


class ModelError(RuntimeError):
    """Модель не загрузилась."""


_CUDA_ERROR_MARKERS = (
    "libcublas",
    "libcudnn",
    "cublas",
    "cudnn",
    "cuda",
    "no kernel image",
    "device-side",
)


def is_cuda_runtime_error(error: BaseException | str) -> bool:
    """Похоже ли сообщение на отказ CUDA-библиотек во время инференса.

    CTranslate2 может успешно загрузить модель на GPU и упасть только на
    первом вычислении, например: «Library libcublas.so.12 is not found».
    """
    text = str(error).lower()
    return any(marker in text for marker in _CUDA_ERROR_MARKERS)


@dataclass
class LoadedModel:
    model: Any
    name: str
    device: str
    compute_type: str
    fallback_reason: str | None = None


def _writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".write-test-"):
            pass
        return True
    except OSError:
        return False


def cache_root(explicit: str | Path | None = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not _writable(path):
            raise ModelError(f"каталог кэша недоступен для записи: {path}")
        return path

    from_env = os.environ.get("TRANSCRIBER_CACHE")
    if from_env:
        path = Path(from_env).expanduser()
        if not _writable(path):
            raise ModelError(f"каталог кэша недоступен для записи: {path}")
        return path

    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    preferred = base / "transcriber"
    if _writable(preferred):
        return preferred

    # Домашний каталог может быть read-only (например, в песочнице).
    fallback = Path(tempfile.gettempdir()) / "transcriber-cache"
    if _writable(fallback):
        return fallback

    raise ModelError(
        "некуда скачивать модели: ~/.cache и /tmp недоступны для записи. "
        "Укажите --cache-dir"
    )


def _configure_huggingface(cache: Path, offline: bool) -> None:
    """Держим кэш моделей внутри каталога проекта/пользователя.

    Переменные нужно выставить до импорта faster-whisper: он импортирует
    huggingface_hub, который читает их при загрузке.
    """
    hf_home = cache / "hf"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    # onnxruntime (VAD) иначе пишет в stderr про telemetry device ID.
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001 — CUDA может падать по-разному
        return False


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if cuda_available() else "cpu"
    if requested == "cuda" and not cuda_available():
        raise ModelError(
            "CUDA не найдена. Запустите с --device cpu или установите "
            "CUDA-библиотеки для CTranslate2."
        )
    return requested


def load_model(
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
    compute_type: str | None = None,
    cache_dir: str | Path | None = None,
    cpu_threads: int = 0,
    offline: bool = False,
) -> LoadedModel:
    cache = cache_root(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    _configure_huggingface(cache, offline)

    # Импорт внутри функции: переменные окружения выше должны быть уже заданы.
    from faster_whisper import WhisperModel

    actual_device = resolve_device(device)
    actual_compute = compute_type
    if not actual_compute or actual_compute == "auto":
        actual_compute = "float16" if actual_device == "cuda" else "int8"

    kwargs: dict[str, Any] = {
        "device": actual_device,
        "compute_type": actual_compute,
        "download_root": str(cache / "models"),
        "cpu_threads": cpu_threads or 0,
    }

    try:
        model = WhisperModel(model_name, **kwargs)
        return LoadedModel(model, model_name, actual_device, actual_compute)
    except Exception as first_error:
        if actual_device != "cuda":
            raise ModelError(
                f"не удалось загрузить модель «{model_name}»: {first_error}"
            ) from first_error

        # CUDA может быть видна, но не хватать cuDNN/cuBLAS. Не падаем,
        # а пробуем CPU — для коротких записей это часто приемлемо.
        reason = (
            f"CUDA недоступна для запуска ({type(first_error).__name__}); "
            "переключаюсь на CPU"
        )
        kwargs.update(device="cpu", compute_type="int8")
        try:
            model = WhisperModel(model_name, **kwargs)
        except Exception as cpu_error:
            raise ModelError(
                f"не удалось загрузить модель «{model_name}» ни на GPU, "
                f"ни на CPU: {cpu_error}"
            ) from cpu_error

        return LoadedModel(model, model_name, "cpu", "int8", reason)
