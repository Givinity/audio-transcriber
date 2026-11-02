from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000

DENOISE_FILTERS: dict[str, str] = {
    # Мягкая обработка узкополосной телефонной записи: убираем низкочастотный
    # гул, шум и выравниваем громкость.
    "phone": "highpass=f=120,afftdn=nf=-20,dynaudnorm=f=200:g=15",
    # Более агрессивный вариант для очень шумных записей.
    "strong": "highpass=f=100,afftdn=nf=-25,dynaudnorm=f=150:g=20",
}


class AudioError(RuntimeError):
    """Не удалось прочитать или декодировать аудио."""


def require_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise AudioError("ffmpeg не найден. Установите его: sudo apt install ffmpeg")
    return executable


def decode_audio(
    path: str | Path,
    denoise: str = "none",
    ffmpeg: str | None = None,
) -> tuple[np.ndarray, float]:
    """Декодирует файл в моно float32 16 кГц.

    Возвращает (сэмплы, длительность в секундах). Аудио декодируется через
    ffmpeg в поток, поэтому временные файлы не создаются.
    """
    if denoise not in {"none", *DENOISE_FILTERS}:
        raise ValueError(f"неизвестный режим шумоподавления: {denoise}")

    ffmpeg = ffmpeg or require_ffmpeg()
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"файл не найден: {path}")

    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
    ]
    if denoise != "none":
        command += ["-af", DENOISE_FILTERS[denoise]]
    command += ["-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]

    try:
        process = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:
        raise AudioError(f"не удалось запустить ffmpeg: {exc}") from exc

    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        tail = "\n".join(stderr.splitlines()[-5:])
        raise AudioError(
            f"ffmpeg не смог декодировать «{path.name}»"
            + (f":\n{tail}" if tail else "")
        )

    samples = np.frombuffer(process.stdout, dtype=np.float32).copy()
    if samples.size == 0:
        raise AudioError(f"в «{path.name}» нет аудиодорожки или она пустая")

    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak < 1e-5:
        raise AudioError(f"«{path.name}»: дорожка почти беззвучная")

    return samples, samples.size / SAMPLE_RATE
