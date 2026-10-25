#!/usr/bin/env python3
"""Отдельный веб-редактор промптов.

Запуск:
    python scripts/prompt_editor.py

Открывает локальную страницу со списком файлов из ``prompts/``, просмотром
Markdown и редактированием. Сервер использует только стандартную библиотеку,
поэтому ничего дополнительно устанавливать не нужно.

Примеры:
    python scripts/prompt_editor.py --port 9000
    python scripts/prompt_editor.py --dir prompts --no-browser
    python scripts/prompt_editor.py --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sipbot.prompt_ui_server import start_prompt_editor  # noqa: E402
from sipbot.prompts import PromptError, PromptStore  # noqa: E402

DEFAULT_DIR = ROOT / "prompts"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Веб-редактор промптов (локальный, без внешних зависимостей)"
    )
    parser.add_argument(
        "--dir",
        default=str(DEFAULT_DIR),
        help="каталог с промптами (по умолчанию: %(default)s)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="адрес прослушивания (по умолчанию: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8766,
        help="порт (по умолчанию: %(default)s; 0 — свободный порт)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="не открывать браузер автоматически",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="подробный лог HTTP-запросов",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        store = PromptStore(args.dir)
    except (PromptError, OSError) as exc:
        print(f"Не удалось открыть каталог промптов: {exc}", file=sys.stderr)
        return 1

    try:
        server = start_prompt_editor(store, host=args.host, port=args.port)
    except OSError as exc:
        print(
            f"Не удалось запустить сервер на {args.host}:{args.port}: {exc}",
            file=sys.stderr,
        )
        return 2

    print(f"Редактор промптов: {server.url}")
    print(f"Каталог: {store.root}")
    print("Остановить: Ctrl+C")

    if not args.no_browser:
        try:
            webbrowser.open(server.url)
        except Exception as exc:  # pragma: no cover - зависит от окружения
            print(f"Браузер не открылся автоматически: {exc}", file=sys.stderr)

    try:
        server.wait()
    except KeyboardInterrupt:
        print("\nОстановка редактора…")
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
