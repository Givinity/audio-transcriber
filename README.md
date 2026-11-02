# audio-transcriber

Локальная расшифровка аудио и видео в текст для Linux (Ubuntu/Debian).
Работает на [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — это
Whisper на CTranslate2: быстро, без PyTorch и без облака.

После первого скачивания модели интернет не нужен. Аудио никуда не
отправляется.

## Что умеет

- MP3, WAV, M4A, OGG/Opus, FLAC, AMR, WMA, а также видео (MP4, MKV, WEBM, MOV):
  звук извлекается через ffmpeg.
- Один файл или целая папка, в том числе рекурсивно.
- Автоопределение языка или явное указание (`ru`, `en`, `kk`, ...).
- Вывод в TXT, SRT, VTT и JSON с таймкодами и метаданными.
- Режим `--denoise phone` для узкополосных телефонных записей (8 кГц).
- CPU и CUDA; при проблемах с CUDA автоматически откатывается на CPU.

## Требования

- Ubuntu 22.04/24.04 (или другой Linux с glibc >= 2.35), x86_64.
- Python 3.10+.
- ffmpeg.
- Свободное место под модели: от 75 МБ (tiny) до 3.1 ГБ (large-v3).

```bash
sudo apt update
sudo apt install -y ffmpeg python3-venv
```

## Установка

```bash
cd audio-transcriber
./install.sh
```

Скрипт создаёт `.venv` внутри проекта и ставит пакет в него. Системный Python
не трогается, sudo не нужен (кроме установки ffmpeg/python3-venv).

Проверка:

```bash
./transcriber --version
./transcriber --help
```

Можно поставить и глобально в пользовательский pipx/venv, но проще
пользоваться обёрткой `./transcriber`.

## Быстрый старт

```bash
# Русскоязычный телефонный звонок
./transcriber "2025-11-28 09-29-49.mp3" --language ru --denoise phone
```

Рядом с файлом появятся `2025-11-28 09-29-49.txt` и (если попросить) `.srt`.

Папка с записями:

```bash
./transcriber ~/Записи/звонки \
  --output-dir ~/Расшифровки \
  --formats txt,srt \
  --language ru \
  --denoise phone
```

Один файл в stdout:

```bash
./transcriber call.mp3 --stdout > call.txt
```

## Полезные примеры

```bash
# Быстро и грубо, чтобы понять, о чём запись
./transcriber call.mp3 -m tiny -l ru

# Точнее, но медленнее
./transcriber interview.m4a -m medium -l ru

# Строки с таймкодами в TXT
./transcriber lecture.mp3 --timestamps

# JSON с таймкодами слов (для монтажа/субтитров)
./transcriber call.mp3 -f json --word-timestamps

# Подсказать термины и имена
./transcriber meetup.m4a --prompt "Kazakhstan, сайт, законопроект, акимат"

# Посмотреть, что будет обработано, ничего не запуская
./transcriber ~/Записи --dry-run -f txt,srt

# Пропустить уже расшифрованное
./transcriber ~/Записи -o ~/Расшифровки --skip-existing
```

## Опции

| Опция | По умолчанию | Описание |
|---|---|---|
| `-o, --output-dir ПАПКА` | рядом с файлом | Куда писать результаты |
| `-f, --formats СПИСОК` | `txt` | `txt,srt,vtt,json` через запятую |
| `-l, --language ЯЗЫК` | `auto` | Код языка или `auto` |
| `-m, --model МОДЕЛЬ` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3`, `distil-large-v3` или путь |
| `--device` | `auto` | `auto`, `cpu`, `cuda` |
| `--compute-type` | `auto` | `int8` на CPU, `float16` на GPU; можно задать вручную |
| `--beam-size N` | `5` | Больше — обычно точнее и медленнее |
| `--denoise` | `none` | `none`, `phone`, `strong` |
| `--prompt ТЕКСТ` | — | Подсказка с терминами/именами |
| `--word-timestamps` | выкл. | Таймкоды слов в JSON |
| `--no-vad` | VAD вкл. | Отключить вырезание тишины |
| `--timestamps` | выкл. | Таймкоды в TXT |
| `--max-line-width N` | `42` | Перенос строк в SRT/VTT |
| `--no-recursive` | рекурсия вкл. | Не обходить вложенные папки |
| `--skip-existing` | выкл. | Пропускать готовые файлы |
| `--dry-run` | выкл. | Показать план |
| `--stdout` | выкл. | Один файл, один формат — в stdout |
| `--cache-dir ПАПКА` | `~/.cache/transcriber` | Кэш моделей |
| `--cpu-threads N` | `0` (авто) | Число CPU-потоков |
| `--offline` | выкл. | Не скачивать модель |
| `-q, --quiet` | выкл. | Без прогресса |
| `--verbose` | выкл. | Параметры запуска |

## Выходные файлы

По умолчанию — рядом с исходником, с тем же именем:

- `call.txt` — читаемый текст, абзацы по паузам;
- `call.srt` — субтитры;
- `call.vtt` — субтитры для веба;
- `call.json` — метаданные и сегменты:

```json
{
  "file": "/home/user/call.mp3",
  "duration": 96.624,
  "language": "ru",
  "language_probability": 1.0,
  "model": "small",
  "device": "cpu",
  "compute_type": "int8",
  "created_at": "2026-11-02T07:00:00+00:00",
  "settings": {"task": "transcribe", "beam_size": 5, "vad": true,
               "denoise": "phone", "word_timestamps": false},
  "segments": [
    {"start": 0.24, "end": 3.24, "text": "..."}
  ]
}
```

С `--output-dir` одинаковые имена не затираются: второй `call.mp3` даст
`call-2.txt` и `call-2.srt`.

## Модели и скорость

| Модель | Размер | Ориентир на CPU (int8) |
|---|---|---|
| tiny | ~75 МБ | ~20–30× быстрее реального времени |
| base | ~145 МБ | ~15× |
| small | ~484 МБ | ~8–12× |
| medium | ~1.5 ГБ | ~2–3× |
| large-v3 | ~3.1 ГБ | ~1–2× |

Замер на этой машине: 97-секундный звонок, 8 кГц, `small` — около 9 секунд,
`medium` — около 45–60 секунд, `large-v3` — около 1,5–2 минут.

Для русского телефонного разговора `small` даёт черновик, `medium` — заметно
лучше. Если запись короткая и важна точность, берите `medium`; если важна
скорость — `small`.

Модель скачивается при первом запуске в `~/.cache/transcriber` (или в
`--cache-dir`). Если домашний каталог недоступен для записи, используется
`/tmp/transcriber-cache`. Повторные запуски работают офлайн.

## GPU

`--device auto` использует CUDA, если CTranslate2 её видит. Для CTranslate2 4.x
нужны CUDA 12 и cuDNN 9; одной видеокарты недостаточно. `auto` проверяет
наличие устройства, но не наличие библиотек: модель может загрузиться на GPU
и упасть на первом вычислении с `libcublas.so.12 is not found`. В этом случае
программа предупреждает и автоматически перезапускает файл на CPU, а
следующие файлы обрабатывает уже на CPU.

```bash
./transcriber call.mp3 --device cuda
./transcriber call.mp3 --device cuda --compute-type float16
```

Если GPU не нужен или библиотеки не установлены, надёжнее сразу указать
`--device cpu` — не будет лишней попытки и предупреждения.

## Приватность

- Аудио не отправляется на серверы.
- Сеть нужна только для скачивания модели с Hugging Face.
- После скачивания можно работать с `--offline`.
- Кэш моделей — обычные файлы в `~/.cache/transcriber`; их можно удалить.

## Точность и ограничения

- Whisper — не «идеальный стенограф». Имена, суммы, аббревиатуры и плохая
  связь остаются камнем преткновения; проверяйте важные детали.
- Узкополосные звонки (8 кГц) распознаются хуже, чем широкая полоса.
  Помогают `--denoise phone`, `--language ru` и модель `medium`.
- VAD вырезает длинную тишину; для записей с очень короткими репликами можно
  попробовать `--no-vad`.
- Диаризации (кто говорит) нет: это отдельная задача.
- Длинные файлы декодируются в память целиком: примерно 230 МБ на час
  аудио в float32. Для нескольких часов лучше запускать по одному файлу.

### CUDA: `libcublas.so.12 is not found`

CTranslate2 4.x ищет CUDA 12 и cuDNN 9. Если их нет, GPU-запуск падает уже
после загрузки модели. Программа поймает это на первом файле и продолжит на
CPU; чтобы не тратить время на GPU, используйте `--device cpu`.

Если GPU нужен, проверьте:

```bash
nvidia-smi
ldconfig -p | grep -E 'libcublas|libcudnn'
```

и установите CUDA 12 + cuDNN 9, совместимые с вашим драйвером.

## Коды возврата

- `0` — всё обработано;
- `1` — были ошибки чтения/записи или часть файлов не обработана;
- `2` — неверные аргументы;
- `130` — прервано Ctrl+C.

## Разработка

```bash
./install.sh --dev
.venv/bin/python -m pytest
```

Структура:

```
src/transcriber/
  cli.py        аргументы, прогресс, сценарий запуска
  audio.py      ffmpeg → float32 16 кГц, шумоподавление
  models.py     загрузка faster-whisper, CPU/CUDA, кэш
  pipeline.py   распознавание и метаданные
  output.py     TXT/SRT/VTT/JSON, атомарная запись
  files.py      сбор файлов и сортировка
```

## Лицензия

MIT. Whisper и faster-whisper распространяются под своими лицензиями (MIT).