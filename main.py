"""CLI тестового SIP-клиента для задач голосового бота.

Примеры:

    python main.py register
    python main.py listen --record in.wav --bot echo
    python main.py dial 1001 --play greeting.wav --record out.wav --duration 20
    python main.py dial 1001 --bot tone --duration 10
    python main.py incoming --bot placeholder --record-dir ./recordings --max-lines 16
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional

from sipbot.bots import build_bot
from sipbot.config import Settings
from sipbot.incoming import IncomingDispatcher
from sipbot.media import MediaBridge, Recorder
from sipbot.phone import SipPhone

logger = logging.getLogger("sipbot")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env(args.env)
    settings.validate()
    return settings


def start_phone(settings: Settings, timeout: float) -> SipPhone:
    phone = SipPhone(settings)
    phone.start()
    status = phone.wait_registered(timeout)
    logger.info("Статус регистрации: %s", status.name)
    if status.name != "REGISTERED":
        phone.stop()
        raise SystemExit(
            "Регистрация не удалась. Проверьте SIP_SERVER / SIP_USER / SIP_PASS, "
            "порт UDP и SIP_MY_IP."
        )
    return phone


def run_call(
    phone: SipPhone,
    call,
    args: argparse.Namespace,
    bot_spec: Optional[str],
) -> None:
    """Поднимает аудио-мост и ждёт завершения звонка."""
    recorder = Recorder(args.record) if getattr(args, "record", None) else None
    bot = build_bot(bot_spec)
    bridge = MediaBridge(call, bot=bot, recorder=recorder)
    bridge.start()
    logger.info(
        "Аудио-мост запущен: бот=%s, запись=%s, длительность=%s",
        bot_spec or "none",
        args.record or "-",
        args.duration if args.duration > 0 else "до отбоя",
    )
    try:
        reason = phone.wait_ended(call, args.duration)
        logger.info("Звонок завершён (%s)", reason)
    finally:
        bridge.stop()
        if recorder is not None:
            recorder.close()
        if bot is not None:
            bot.close()
        logger.info("Статистика обмена: %s", bridge.stats())


def cmd_register(settings: Settings, args: argparse.Namespace) -> int:
    phone = start_phone(settings, args.timeout)
    try:
        if args.hold > 0:
            logger.info("Держу регистрацию %.1f с...", args.hold)
            time.sleep(args.hold)
        logger.info("Итоговый статус: %s", phone.status().name)
    finally:
        phone.stop()
    return 0


def cmd_listen(settings: Settings, args: argparse.Namespace) -> int:
    phone = start_phone(settings, 20.0)
    deadline = time.monotonic() + args.timeout if args.timeout > 0 else None
    try:
        logger.info(
            "Ожидание входящих звонков%s...",
            f" ({args.timeout:.0f} с)" if deadline else "",
        )
        while deadline is None or time.monotonic() < deadline:
            call = phone.next_call(timeout=1.0)
            if call is None:
                continue
            if not phone.answer(call):
                continue
            try:
                run_call(phone, call, args, args.bot)
            finally:
                phone.hangup(call)
            logger.info("Готов к следующему звонку")
    finally:
        phone.stop()
    return 0


def cmd_incoming(settings: Settings, args: argparse.Namespace) -> int:
    """Многоканальный приём входящих звонков."""
    if args.max_lines is not None:
        settings = settings.__class__(
            server=settings.server,
            username=settings.username,
            password=settings.password,
            my_ip=settings.my_ip,
            port=settings.port,
            sip_port=settings.sip_port,
            rtp_port_low=settings.rtp_port_low,
            rtp_port_high=settings.rtp_port_high,
            max_incoming_lines=args.max_lines,
        )
        settings.validate()

    phone = SipPhone(settings)
    dispatcher = IncomingDispatcher(
        phone=phone,
        settings=settings,
        bot_factory=lambda: build_bot(args.bot),
        record_dir=args.record_dir,
    )
    phone.start(on_incoming=dispatcher.handle_incoming)
    status = phone.wait_registered(20.0)
    logger.info("Статус регистрации: %s", status.name)
    if status.name != "REGISTERED":
        phone.stop()
        dispatcher.shutdown()
        raise SystemExit("Регистрация не удалась")

    logger.info(
        "Многоканальный приём входящих: bot=%s max_lines=%d record_dir=%s",
        args.bot,
        settings.max_incoming_lines,
        args.record_dir,
    )
    try:
        while True:
            time.sleep(1)
            logger.debug(
                "Активных линий: %d/%d",
                dispatcher.active_count,
                settings.max_incoming_lines,
            )
    except KeyboardInterrupt:
        logger.info("Завершение работы...")
    finally:
        dispatcher.shutdown()
        phone.stop()
    return 0


def cmd_dial(settings: Settings, args: argparse.Namespace) -> int:
    phone = start_phone(settings, 20.0)
    try:
        call = phone.dial(args.target)
        if not phone.wait_answered(call, args.timeout):
            state = getattr(getattr(call, "state", None), "name", "?")
            logger.error(
                "Звонок не состоялся: state=%s (ожидание ответа до %.0f с)",
                state,
                args.timeout,
            )
            phone.hangup(call)
            return 2
        bot_spec = f"wav:{args.play}" if args.play else args.bot
        run_call(phone, call, args, bot_spec)
        phone.hangup(call)
    finally:
        phone.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Тестовый SIP-клиент (pyVoIP) для нужд голосового бота"
    )
    parser.add_argument(
        "--env", default=".env", help="путь к файлу .env (по умолчанию .env)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="отладочные логи")
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register", help="только регистрация на SIP-сервере")
    reg.add_argument(
        "--timeout", type=float, default=20.0, help="ожидание регистрации, с"
    )
    reg.add_argument(
        "--hold", type=float, default=0.0, help="сколько держать регистрацию, с"
    )
    reg.set_defaults(func=cmd_register)

    listen = sub.add_parser("listen", help="принимать входящие звонки")
    listen.add_argument("--record", help="писать входящий звук в WAV")
    listen.add_argument(
        "--bot",
        default="silence",
        help="обработчик: silence|echo|tone|tone:<Гц>:<мс>|wav:<путь>[:loop]",
    )
    listen.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="общее время ожидания звонков, с (0 = бесконечно)",
    )
    listen.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="максимальная длительность одного звонка, с (0 = до отбоя)",
    )
    listen.set_defaults(func=cmd_listen)

    dial = sub.add_parser("dial", help="исходящий звонок")
    dial.add_argument(
        "target", help="номер/логин; можно sip:1001@host — хост игнорируется"
    )
    dial.add_argument("--play", help="WAV-файл для проигрывания собеседнику")
    dial.add_argument(
        "--bot",
        default=None,
        help="обработчик входящего звука: silence|echo|tone|tone:<Гц>:<мс>|wav:<путь>",
    )
    dial.add_argument("--record", help="писать звук собеседника в WAV")
    dial.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="длительность звонка после ответа, с (0 = до отбоя)",
    )
    dial.add_argument("--timeout", type=float, default=20.0, help="ожидание ответа, с")
    dial.set_defaults(func=cmd_dial)

    incoming = sub.add_parser("incoming", help="многоканальный приём входящих звонков")
    incoming.add_argument(
        "--bot",
        default="placeholder",
        help="обработчик: silence|echo|tone|placeholder|placeholder:<мс>|wav:<путь>[:loop]",
    )
    incoming.add_argument(
        "--record-dir",
        default="recordings",
        help="директория для записи всех входящих звонков",
    )
    incoming.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="переопределить MAX_INCOMING_LINES",
    )
    incoming.set_defaults(func=cmd_incoming)

    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        settings = load_settings(args)
        return args.func(settings, args)
    except ValueError as exc:
        logger.error("Ошибка конфигурации: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Прервано пользователем")
        return 130


if __name__ == "__main__":
    sys.exit(main())
