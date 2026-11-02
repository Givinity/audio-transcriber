from __future__ import annotations

import re
from pathlib import Path

# Расширения, которые подхватываются при обходе папок.
# Явно указанный файл принимается с любым расширением — решение остаётся за ffmpeg.
AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".amr",
    ".avi",
    ".flac",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
    ".wv",
}


def natural_key(path: Path) -> list[object]:
    """Ключ сортировки, при котором «10» идёт после «9», а не перед «1»."""
    parts = re.split(r"(\d+)", str(path).lower())
    return [int(part) if part.isdigit() else part for part in parts]


def collect_files(
    inputs: list[str], recursive: bool = True
) -> tuple[list[Path], list[str]]:
    """Разворачивает файлы и папки в плоский список файлов.

    Возвращает (файлы, ошибки). Ошибки — это несуществующие пути и пути,
    которые не являются ни файлом, ни папкой.
    """
    found: list[Path] = []
    errors: list[str] = []
    seen: set[str] = set()

    for raw in inputs:
        path = Path(raw).expanduser()
        try:
            path = path.resolve()
        except OSError:
            pass

        if not path.exists():
            errors.append(f"путь не найден: {raw}")
            continue

        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            for file_path in iterator:
                if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS:
                    key = str(file_path)
                    if key not in seen:
                        seen.add(key)
                        found.append(file_path)
        elif path.is_file():
            key = str(path)
            if key not in seen:
                seen.add(key)
                found.append(path)
        else:
            errors.append(f"это не файл и не папка: {raw}")

    found.sort(key=natural_key)
    return found, errors
