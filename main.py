"""CLI тестового SIP-клиента для задач голосового бота.

Примеры:

    python main.py register
    python main.py listen --record in.wav --bot echo
    python main.py dial 1001 --play greeting.wav --record out.wav --duration 20
    python main.py dial 1001 --bot tone --duration 10
    python main.py incoming --record-dir ./recordings --max-lines 16
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from sipbot.ai_client import AiClient
from sipbot.billing import BillingClient
from sipbot.bots import Bot, build_bot
from sipbot.config import Settings
from sipbot.dialog import DialogBot, DialogConfig
from sipbot.incoming import IncomingDispatcher
from sipbot.media import MediaBridge, Recorder
from sipbot.phone import SipPhone
from sipbot.prompts import PromptStore
from sipbot.transcript import SYSTEM, TranscriptStore
from sipbot.transcribing_bot import TranscribingBot
from sipbot.tts import TtsService
from sipbot.ui_server import start_ui_server

logger = logging.getLogger("sipbot")
PROJECT_ROOT = Path(__file__).resolve().parent


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


def load_dialog_prompt() -> str:
    """Собрать system prompt из prompts/ по SYSTEM_PROMPT_FILES."""
    store = PromptStore(PROJECT_ROOT / "prompts")
    bundle = store.load_system_prompt_from_env()
    logger.info(
        "System prompt: %s (%.1f КБ)",
        ", ".join(bundle.names),
        bundle.size / 1024.0,
    )
    return bundle.content


def build_dialog_factory(
    ai: AiClient,
    store: Optional[TranscriptStore],
    system_prompt: str,
    config: DialogConfig,
    billing: Optional[BillingClient] = None,
) -> Callable[[], DialogBot]:
    """Создать фабрику диалоговых ботов с общим TTS-кэшем.

    Приветствие синтезируется один раз при старте (и кэшируется на диск), а не
    на каждом входящем звонке. Если TTS недоступен, клиент продолжает работать:
    бот ответит на реплики, а в лог попадёт ошибка синтеза приветствия.
    """
    tts = TtsService(ai, cache_dir=config.tts_cache_dir)
    greeting_audio = b""
    if config.greeting:
        try:
            logger.info("Синтез приветствия: %r", config.greeting[:80])
            greeting_audio = tts.preload(config.greeting)
            logger.info(
                "Приветствие готово: %.1f с аудио",
                len(greeting_audio) / 8000.0,
            )
        except Exception:  # noqa: BLE001 - без TTS клиент всё равно полезен
            logger.exception(
                "Не удалось синтезировать приветствие; проверьте TTS_MODEL, "
                "TTS_VOICE и доступность OPENAI_BASE_URL"
            )

    if config.no_input_prompt and config.no_input_prompt_ms > 0:
        try:
            tts.preload(config.no_input_prompt)
            logger.info("No-input подсказка синтезирована заранее")
        except Exception:  # noqa: BLE001 - подсказка не критична для звонка
            logger.exception("Не удалось синтезировать no-input подсказку")

    def factory() -> DialogBot:
        return DialogBot(
            ai,
            system_prompt,
            store=store,
            config=config,
            tts=tts,
            greeting_audio=greeting_audio,
            billing=billing,
        )

    return factory


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

    dialog_config = DialogConfig.from_env()
    raw_bot = (args.bot or "auto").strip()
    requested_bot = raw_bot.lower() if raw_bot.lower() in ("auto", "ai") else raw_bot
    store = TranscriptStore() if args.ui else None
    ai = None if args.no_asr else AiClient.from_env()
    billing = None if args.no_asr else BillingClient.from_env()
    dialog_factory = None
    chosen_bot = requested_bot

    if requested_bot in ("auto", "ai"):
        if args.no_asr:
            if requested_bot == "ai":
                logger.error("--bot ai несовместим с --no-asr: диалогу нужен ASR")
                return 2
            logger.warning("AI-бот отключён ключом --no-asr, использую placeholder")
            chosen_bot = "placeholder"
        elif ai is None:
            if requested_bot == "ai":
                logger.error(
                    "AI-бот не настроен: задайте OPENAI_BASE_URL и OPENAI_API_KEY"
                )
                return 2
            logger.warning(
                "AI не настроен (OPENAI_BASE_URL/OPENAI_API_KEY), "
                "использую placeholder"
            )
            chosen_bot = "placeholder"
        else:
            try:
                system_prompt = load_dialog_prompt()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Не удалось загрузить system prompt")
                logger.error("Проверьте SYSTEM_PROMPT_FILES и каталог prompts/: %s", exc)
                return 2
            if billing is None:
                logger.warning(
                    "Биллинг не настроен (BILLING_BASE_URL/BILLING_TOKEN); "
                    "бот сможет предложить оператора, но не назовёт баланс"
                )
            else:
                logger.info("Биллинг подключён: %s", billing.base_url)
            dialog_factory = build_dialog_factory(
                ai, store, system_prompt, dialog_config, billing=billing
            )
            chosen_bot = "ai"

    ui_server = None
    if args.ui:
        assert store is not None
        if ai is None:
            reason = (
                "ASR отключён ключом --no-asr"
                if args.no_asr
                else "ASR не настроен: задайте OPENAI_BASE_URL и OPENAI_API_KEY в .env"
            )
            logger.warning(reason)
            store.publish(SYSTEM, reason)
        elif dialog_factory is not None:
            logger.info(
                "AI-бот: ASR=%s, LLM=%s, TTS=%s/%s",
                ai.asr_model,
                ai.llm_model,
                ai.tts_model,
                ai.tts_voice,
            )
        else:
            logger.info("ASR: модель=%s язык=%s", ai.asr_model, ai.asr_language)
        try:
            ui_server = start_ui_server(store, args.ui_host, args.ui_port)
        except OSError as exc:
            logger.error(
                "Не удалось запустить веб-интерфейс на %s:%d: %s",
                args.ui_host,
                args.ui_port,
                exc,
            )
            return 2

    def make_bot() -> Optional[Bot]:
        if dialog_factory is not None:
            return dialog_factory()
        bot = build_bot(chosen_bot)
        if store is not None:
            # В режиме только-UI оставляем прежнюю схему: бот отвечает звуком,
            # а распознавание идёт отдельной обёрткой.
            return TranscribingBot(bot, store, ai, vad=dialog_config.make_vad())
        return bot

    phone = SipPhone(settings)
    dispatcher = IncomingDispatcher(
        phone=phone,
        settings=settings,
        bot_factory=make_bot,
        record_dir=args.record_dir,
    )
    try:
        phone.start(on_incoming=dispatcher.handle_incoming)
        status = phone.wait_registered(20.0)
        logger.info("Статус регистрации: %s", status.name)
        if status.name != "REGISTERED":
            raise SystemExit("Регистрация не удалась")

        logger.info(
            "Многоканальный приём входящих: bot=%s (запрошен %s), "
            "max_lines=%d record_dir=%s",
            chosen_bot,
            requested_bot,
            settings.max_incoming_lines,
            args.record_dir,
        )
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
        if ui_server is not None:
            ui_server.stop()
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
        default="auto",
        help=(
            "обработчик: auto|ai|silence|echo|tone|placeholder|"
            "placeholder:<мс>|wav:<путь>[:loop]. auto выбирает ai, "
            "если настроены ASR/LLM/TTS, иначе placeholder"
        ),
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
    incoming.add_argument(
        "--ui",
        action="store_true",
        help="запустить веб-интерфейс с живой расшифровкой разговора",
    )
    incoming.add_argument(
        "--ui-host",
        default="127.0.0.1",
        help="адрес веб-интерфейса (по умолчанию 127.0.0.1)",
    )
    incoming.add_argument(
        "--ui-port",
        type=int,
        default=8765,
        help="порт веб-интерфейса (по умолчанию 8765)",
    )
    incoming.add_argument(
        "--no-asr",
        action="store_true",
        help="отключить ASR и AI-бота (UI покажет только события звонка)",
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
