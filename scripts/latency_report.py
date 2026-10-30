"""Отчёт по задержкам ASR → LLM → TTS из лога или JSONL.

Примеры:
    python scripts/latency_report.py logs/call.log --details
    python scripts/latency_report.py metrics.jsonl --last 20 --json
    type logs\\call.log | python scripts/latency_report.py -

Источник определяется по содержимому: JSONL (строки с ``{``) или текстовый
лог с репликами «Задержка ответа [...]». Битые строки просто пропускаются.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sipbot.metrics import (  # noqa: E402
    format_report,
    load_samples,
    parse_jsonl,
    parse_log_text,
    summarize,
)


def _read_stdin() -> list:
    text = sys.stdin.read()
    if text.lstrip().startswith("{"):
        return parse_jsonl(text)
    return parse_log_text(text)


def _load(path: str) -> list:
    if path == "-":
        return _read_stdin()
    source = Path(path)
    if not source.is_file():
        raise SystemExit(f"Файл не найден: {source}")
    return load_samples(source)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Статистика задержек реплик (лог или metrics.jsonl)",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["-"],
        help="файлы логов/JSONL; '-' или пусто — stdin",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=0,
        metavar="N",
        help="учесть только последние N реплик",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="показать разбивку LLM по раундам и TTS по предложениям",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="вывести агрегаты в JSON",
    )
    return parser


def main(argv: list = None) -> int:
    args = build_parser().parse_args(argv)
    paths = args.paths or ["-"]

    samples = []
    for path in paths:
        samples.extend(_load(path))

    if args.last > 0:
        samples = samples[-args.last :]

    if not samples:
        print(
            "Метрик задержек не найдено. Включите запись "
            "(main.py dial ... --log-file call.log) или разберите файл целиком.",
            file=sys.stderr,
        )
        return 1

    summary = summarize(samples)
    if args.as_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(format_report(summary, details=args.details))
    return 0


if __name__ == "__main__":
    sys.exit(main())
