#!/usr/bin/env bash
# Установка в локальное окружение проекта. Ничего не ставит в систему.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
WITH_DEV=0

if [[ "${1:-}" == "--dev" ]]; then
    WITH_DEV=1
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Не найден $PYTHON. Установите Python 3.10+." >&2
    exit 1
fi

if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "Нужен Python 3.10 или новее. Сейчас: $("$PYTHON" --version 2>&1)" >&2
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Не найден ffmpeg. Установите:" >&2
    echo "  sudo apt update && sudo apt install -y ffmpeg" >&2
    exit 1
fi

if [[ ! -d "$HERE/.venv" ]]; then
    echo "Создаю виртуальное окружение .venv ..."
    if ! "$PYTHON" -m venv "$HERE/.venv"; then
        echo "Не удалось создать venv. Установите пакет python3-venv:" >&2
        echo "  sudo apt install -y python3-venv" >&2
        exit 1
    fi
fi

"$HERE/.venv/bin/python" -m pip install --upgrade pip setuptools wheel

if [[ "$WITH_DEV" == "1" ]]; then
    "$HERE/.venv/bin/python" -m pip install -e "$HERE[dev]"
else
    "$HERE/.venv/bin/python" -m pip install -e "$HERE"
fi

chmod +x "$HERE/transcriber" 2>/dev/null || true

echo
echo "Готово. Запуск:"
echo "  $HERE/transcriber --help"
echo "  $HERE/transcriber запись.mp3 -l ru"
echo
echo "Модель скачается при первом запуске в ~/.cache/transcriber."