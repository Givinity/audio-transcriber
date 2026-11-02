from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

from . import __version__
from .audio import AudioError
from .files import collect_files
from .models import (
    DEFAULT_MODEL,
    MODEL_HINTS,
    ModelError,
    is_cuda_runtime_error,
    load_model,
)
from .output import (
    FORMATS,
    base_outputs,
    format_duration,
    plan_outputs,
    render_json,
    render_srt,
    render_txt,
    render_vtt,
    write_text_atomic,
)

DESCRIPTION = (
    "Локальная расшифровка аудио и видео в текст. "
    "Ничего не отправляется в интернет: после скачивания модели всё "
    "считается на этом компьютере."
)

EPILOG = """\
примеры:
  transcriber call.mp3
  transcriber call.mp3 -l ru --denoise phone
  transcriber ~/Записи -o ~/Расшифровки -f txt,srt -l ru
  transcriber interview.m4a --model medium --timestamps
  transcriber call.mp3 --stdout > call.txt
"""


class Reporter:
    def __init__(
        self,
        quiet: bool = False,
        verbose: bool = False,
        stream: TextIO | None = None,
    ) -> None:
        self.quiet = quiet
        self.verbose = verbose
        self.stream: TextIO = stream or sys.stdout

    def info(self, message: str) -> None:
        if not self.quiet:
            print(message, file=self.stream, flush=True)

    def detail(self, message: str) -> None:
        if self.verbose and not self.quiet:
            print(message, file=self.stream, flush=True)

    def warn(self, message: str) -> None:
        print(f"предупреждение: {message}", file=sys.stderr, flush=True)

    def error(self, message: str) -> None:
        print(f"ошибка: {message}", file=sys.stderr, flush=True)


class Progress:
    """Процент выполнения в stderr, только если это терминал."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled and sys.stderr.isatty()
        self.last_percent = -1

    def __call__(self, done: float, total: float) -> None:
        if not self.enabled:
            return
        percent = 100 if total <= 0 else min(100, int(done / total * 100))
        if percent != self.last_percent:
            self.last_percent = percent
            print(f"\r  {percent:3d}%", end="", file=sys.stderr, flush=True)

    def reset(self) -> None:
        self.last_percent = -1

    def finish(self) -> None:
        if self.enabled and self.last_percent >= 0:
            print(file=sys.stderr, flush=True)


def parse_formats(value: str) -> list[str]:
    items: list[str] = []
    for raw in value.replace(";", ",").replace(" ", ",").split(","):
        item = raw.strip().lower().lstrip(".")
        if not item:
            continue
        if item not in FORMATS:
            raise ValueError(
                f"неизвестный формат «{item}»; доступны: {', '.join(FORMATS)}"
            )
        if item not in items:
            items.append(item)
    if not items:
        raise ValueError("не указан ни один формат вывода")
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcriber",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="ФАЙЛ_ИЛИ_ПАПКА",
        help="аудио/видео файлы или папки с ними",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        metavar="ПАПКА",
        help="куда писать результаты (по умолчанию — рядом с исходником)",
    )
    parser.add_argument(
        "-f",
        "--formats",
        default="txt",
        metavar="СПИСОК",
        help="форматы через запятую: txt, srt, vtt, json (по умолчанию: txt)",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="auto",
        metavar="ЯЗЫК",
        help="код языка (ru, en, kk, ...) или auto для автоопределения",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        metavar="МОДЕЛЬ",
        help=(
            "модель faster-whisper (по умолчанию: small). "
            + "; ".join(f"{name} — {hint}" for name, hint in MODEL_HINTS.items())
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="устройство вычислений (по умолчанию: auto)",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        metavar="ТИП",
        help="int8, float16, float32, ... (по умолчанию: int8 на CPU, float16 на GPU)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        metavar="N",
        help="размер луча декодирования (по умолчанию: 5)",
    )
    parser.add_argument(
        "--denoise",
        choices=("none", "phone", "strong"),
        default="none",
        help="шумоподавление: phone для телефонных записей, strong для очень шумных",
    )
    parser.add_argument(
        "--prompt",
        metavar="ТЕКСТ",
        help="подсказка с терминами/именами (может смещать распознавание)",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="добавить таймкоды слов в JSON",
    )
    parser.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="вырезать длинную тишину перед распознаванием (по умолчанию включено)",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="в TXT писать строки с таймкодами",
    )
    parser.add_argument(
        "--max-line-width",
        type=int,
        default=42,
        metavar="N",
        help="перенос строк в SRT/VTT (по умолчанию: 42)",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="обходить папки рекурсивно (по умолчанию включено)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="пропускать файлы, для которых все результаты уже есть",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="показать, что будет обработано и куда записано",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="вывести результат в stdout (один файл и один формат)",
    )
    parser.add_argument(
        "--cache-dir",
        metavar="ПАПКА",
        help="каталог для моделей (по умолчанию ~/.cache/transcriber)",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        metavar="N",
        help="число CPU-потоков (0 — авто)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="не скачивать модели; использовать уже сохранённые",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="не печатать прогресс и служебные сообщения",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="печатать параметры запуска",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    reporter = Reporter(
        quiet=args.quiet,
        verbose=args.verbose,
        stream=sys.stderr if args.stdout else sys.stdout,
    )

    try:
        formats = parse_formats(args.formats)
    except ValueError as exc:
        reporter.error(str(exc))
        return 2

    if args.beam_size < 1:
        reporter.error("--beam-size должен быть >= 1")
        return 2
    if args.max_line_width < 10:
        reporter.error("--max-line-width должен быть >= 10")
        return 2
    if args.cpu_threads < 0:
        reporter.error("--cpu-threads не может быть отрицательным")
        return 2

    files, path_errors = collect_files(args.inputs, recursive=args.recursive)
    for message in path_errors:
        reporter.error(message)

    if not files:
        reporter.error("не найдено ни одного файла для обработки")
        return 1

    if args.stdout:
        if len(files) != 1:
            reporter.error("--stdout работает только с одним файлом")
            return 2
        if len(formats) != 1:
            reporter.error("--stdout работает только с одним форматом")
            return 2
        if args.output_dir:
            reporter.error("--stdout нельзя сочетать с --output-dir")
            return 2

    reserved: set[Path] = set()
    plans: list[tuple[Path, dict[str, Path] | None]] = []
    for path in files:
        if args.skip_existing and not args.stdout:
            existing = base_outputs(path, formats, args.output_dir)
            if all(output.exists() for output in existing.values()):
                plans.append((path, None))
                continue
        outputs = plan_outputs(path, formats, args.output_dir, reserved)
        plans.append((path, outputs))

    if args.dry_run:
        for path, outputs in plans:
            if outputs is None:
                reporter.info(f"{path} -> пропуск (результаты уже есть)")
            else:
                reporter.info(
                    f"{path} -> {', '.join(str(p) for p in outputs.values())}"
                )
        return 1 if path_errors else 0

    if all(outputs is None for _, outputs in plans):
        reporter.info("Все файлы уже расшифрованы, делать нечего.")
        return 1 if path_errors else 0

    if args.output_dir:
        try:
            Path(args.output_dir).expanduser().mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            reporter.error(f"не удалось создать папку результатов: {exc}")
            return 1

    language = None if args.language.lower() in {"auto", ""} else args.language
    options = _make_options(args, language, reporter)

    reporter.info(
        f"Модель «{args.model}», устройство: {args.device}, "
        f"вычисления: {args.compute_type}"
    )
    try:
        model = load_model(
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            cache_dir=args.cache_dir,
            cpu_threads=args.cpu_threads,
            offline=args.offline,
        )
    except (ModelError, OSError, ValueError) as exc:
        message = str(exc)
        reporter.error(message)
        if args.offline and "cuda" not in message.lower():
            reporter.error(
                "в режиме --offline модель должна быть уже скачана в кэш; "
                "уберите --offline для первой загрузки"
            )
        return 1

    if model.fallback_reason:
        reporter.warn(model.fallback_reason)
    reporter.info(f"Загружено: {model.name} / {model.device} / {model.compute_type}")

    from .pipeline import TranscriptionError

    succeeded = 0
    failed = 0
    skipped = 0
    total = len(plans)

    for index, (path, outputs) in enumerate(plans, start=1):
        if outputs is None:
            reporter.info(f"[{index}/{total}] {path.name} — уже готово, пропускаю")
            skipped += 1
            continue

        reporter.info(f"[{index}/{total}] {path.name}")

        progress = Progress(enabled=not args.quiet)
        try:
            transcript, model = _transcribe_with_fallback(
                path, model, options, progress, reporter
            )
        except (AudioError, TranscriptionError, OSError, ValueError) as exc:
            reporter.error(f"{path.name}: {exc}")
            failed += 1
            continue
        finally:
            progress.finish()

        reporter.info(
            f"  язык: {transcript.language} "
            f"({transcript.language_probability:.0%}), "
            f"длительность: {format_duration(transcript.duration)}"
        )
        if not transcript.segments:
            reporter.warn(f"{path.name}: речь не распознана, результат пустой")

        try:
            if args.stdout:
                fmt = formats[0]
                sys.stdout.write(_render(fmt, transcript, args))
                sys.stdout.flush()
            else:
                for fmt, output_path in outputs.items():
                    write_text_atomic(output_path, _render(fmt, transcript, args))
                reporter.info(
                    "  записано: " + ", ".join(str(p) for p in outputs.values())
                )
        except BrokenPipeError:
            raise
        except OSError as exc:
            reporter.error(f"{path.name}: не удалось записать результат: {exc}")
            failed += 1
            continue

        succeeded += 1

    summary = f"Готово: успешно {succeeded} из {total}"
    if skipped:
        summary += f", пропущено {skipped}"
    if failed:
        summary += f", ошибок {failed}"
    reporter.info(summary)

    if failed or path_errors:
        return 1
    return 0


def _transcribe_with_fallback(
    path: Path,
    model,
    options,
    progress: Progress,
    reporter: Reporter,
):
    """Первый запуск на CUDA может упасть из-за отсутствующих libcublas/libcudnn.

    В этом случае перезагружаем модель на CPU и повторяем файл, а не роняем
    всю пакетную обработку.
    """
    from .pipeline import TranscriptionError, transcribe

    try:
        return transcribe(path, model, options, progress=progress), model
    except TranscriptionError as exc:
        if model.device != "cuda" or not is_cuda_runtime_error(exc):
            raise

        detail = " ".join(str(exc).split())
        if len(detail) > 180:
            detail = detail[:177] + "..."
        reporter.warn(
            f"{path.name}: GPU не заработал ({detail}); переключаюсь на CPU. "
            "Для GPU нужны CUDA 12 и cuDNN 9, либо запустите с --device cpu."
        )
        try:
            cpu_model = load_model(
                model_name=model.name,
                device="cpu",
                compute_type="int8",
                cache_dir=options.cache_dir,
                cpu_threads=options.cpu_threads,
                offline=options.offline,
            )
        except (ModelError, OSError, ValueError) as reload_error:
            raise TranscriptionError(
                f"не удалось переключиться на CPU: {reload_error}"
            ) from reload_error

        progress.reset()
        return transcribe(path, cpu_model, options, progress=progress), cpu_model


def _make_options(args: argparse.Namespace, language: str | None, reporter: Reporter):
    from .pipeline import Options

    options = Options(
        model=args.model,
        language=language,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad=args.vad,
        denoise=args.denoise,
        prompt=args.prompt,
        word_timestamps=args.word_timestamps,
        cache_dir=args.cache_dir,
        cpu_threads=args.cpu_threads,
        offline=args.offline,
    )
    reporter.detail(
        "Параметры: "
        f"language={language or 'auto'}, beam={options.beam_size}, "
        f"vad={options.vad}, denoise={options.denoise}"
    )
    return options


def _render(fmt: str, transcript, args: argparse.Namespace) -> str:
    if fmt == "txt":
        return render_txt(transcript.segments, timestamps=args.timestamps)
    if fmt == "srt":
        return render_srt(transcript.segments, width=args.max_line_width)
    if fmt == "vtt":
        return render_vtt(transcript.segments, width=args.max_line_width)
    if fmt == "json":
        return render_json(transcript.metadata, transcript.segments)
    raise ValueError(f"неизвестный формат: {fmt}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nПрервано пользователем.", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
