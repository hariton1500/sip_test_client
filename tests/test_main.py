"""Тесты выбора бота для ``dial`` и цикла исходящего звонка (без сети)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import main
from sipbot.bots import WavFileBot


class _FakeCall:
    """Минимальный VoIPCall: состояние и Call-ID."""

    def __init__(self, call_id: str = "call-42", state: str = "ANSWERED") -> None:
        self.call_id = call_id
        self.state = SimpleNamespace(name=state)


class _FakeBot:
    def __init__(self, greeting: bytes = b"", end_call: bool = False) -> None:
        self.greeting = greeting
        self.end_call_requested = end_call
        self.started = None
        self.ended = False
        self.closed = False

    def on_call_started(self, caller_id: str, call_id: str):
        self.started = (caller_id, call_id)
        return self.greeting

    def on_call_ended(self) -> None:
        self.ended = True

    def on_frame(self, frame: bytes):
        return None

    def close(self) -> None:
        self.closed = True


class _FakeBridge:
    """Подмена MediaBridge: запоминает инъекцию приветствия и запуск."""

    instances: list = []

    def __init__(self, call, bot=None, recorder=None) -> None:
        self.call = call
        self.bot = bot
        self.recorder = recorder
        self.injected = b""
        self.started = False
        self.stopped = False
        _FakeBridge.instances.append(self)

    def inject(self, data: bytes) -> None:
        self.injected += data

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def stats(self) -> dict:
        return {"frames_in": 0, "frames_out": 0, "dropped": 0, "queued": 0}


class _FakePhone:
    def wait_ended(self, call, duration: float = 0.0) -> str:
        return "ended"


def _args(**overrides) -> SimpleNamespace:
    values = {"play": None, "bot": None, "record": None, "duration": 0.01}
    values.update(overrides)
    return SimpleNamespace(**values)


# ----------------------------------------------------------------------
# _dial_bot: выбор бота
# ----------------------------------------------------------------------
def test_dial_play_overrides_bot_with_wav():
    spec, factory = main._dial_bot(_args(play="greeting.wav", bot="tone"))

    assert spec == "wav:greeting.wav"
    assert factory is None


def test_dial_play_conflicts_with_ai():
    with pytest.raises(ValueError, match="--play"):
        main._dial_bot(_args(play="greeting.wav", bot="ai"))


def test_dial_plain_bot_spec_is_passed_through():
    spec, factory = main._dial_bot(_args(bot="tone:440:200"))

    assert spec == "tone:440:200"
    assert factory is None


def test_dial_without_bot_means_no_bot():
    spec, factory = main._dial_bot(_args())

    assert spec is None
    assert factory is None


def test_dial_ai_without_configuration_is_error(monkeypatch):
    monkeypatch.setattr(main, "build_ai_bot_factory", lambda store=None: None)

    with pytest.raises(ValueError, match="не настроен"):
        main._dial_bot(_args(bot="ai"))


def test_dial_auto_falls_back_to_placeholder(monkeypatch):
    monkeypatch.setattr(main, "build_ai_bot_factory", lambda store=None: None)

    spec, factory = main._dial_bot(_args(bot="auto"))

    assert spec == "placeholder"
    assert factory is None


def test_dial_auto_uses_ai_factory_when_configured(monkeypatch):
    factory = lambda: _FakeBot()  # noqa: E731
    monkeypatch.setattr(main, "build_ai_bot_factory", lambda store=None: factory)

    spec, resolved = main._dial_bot(_args(bot="AUTO"))

    assert spec == "ai"
    assert resolved is factory


# ----------------------------------------------------------------------
# wait_call_end
# ----------------------------------------------------------------------
def test_wait_call_end_returns_ended_when_call_finished():
    assert main.wait_call_end(_FakeCall(state="ENDED"), duration=0) == "ended"


def test_wait_call_end_times_out():
    assert main.wait_call_end(_FakeCall(), duration=0.01) == "timeout"


def test_wait_call_end_waits_for_queued_audio_then_hangs_up():
    bot = _FakeBot(end_call=True)
    bridge = SimpleNamespace(stats=lambda: {"queued": 0})

    assert main.wait_call_end(_FakeCall(), duration=0, bot=bot, bridge=bridge) == "bot"


def test_wait_call_end_does_not_hang_up_while_audio_queued():
    bot = _FakeBot(end_call=True)
    bridge = SimpleNamespace(stats=lambda: {"queued": 3})

    result = main.wait_call_end(_FakeCall(), duration=0.01, bot=bot, bridge=bridge)

    assert result == "timeout"


# ----------------------------------------------------------------------
# run_call: приветствие, завершение, освобождение ресурсов
# ----------------------------------------------------------------------
def test_run_call_injects_greeting_and_closes_bot(monkeypatch):
    _FakeBridge.instances.clear()
    monkeypatch.setattr(main, "MediaBridge", _FakeBridge)
    bot = _FakeBot(greeting=b"\x80" * 320)

    main.run_call(
        _FakePhone(),
        _FakeCall(),
        _args(),
        "ai",
        bot_factory=lambda: bot,
        caller_id="+79780481885",
    )

    bridge = _FakeBridge.instances[-1]
    assert bot.started == ("+79780481885", "call-42")
    assert bridge.injected == b"\x80" * 320
    assert bridge.started and bridge.stopped
    assert bot.ended and bot.closed


def test_run_call_survives_broken_bot_lifecycle(monkeypatch):
    """Падение on_call_started/ended не должно ломать звонок и close()."""

    class _BrokenBot(_FakeBot):
        def on_call_started(self, caller_id, call_id):
            raise RuntimeError("boom")

        def on_call_ended(self):
            raise RuntimeError("boom")

    _FakeBridge.instances.clear()
    monkeypatch.setattr(main, "MediaBridge", _FakeBridge)
    bot = _BrokenBot()

    main.run_call(_FakePhone(), _FakeCall(), _args(), "ai", bot_factory=lambda: bot)

    bridge = _FakeBridge.instances[-1]
    assert bridge.injected == b""
    assert bridge.stopped
    assert bot.closed


def test_wav_bot_has_lifecycle_hooks():
    """run_call зовёт on_call_started/on_call_ended у любого бота."""
    assert hasattr(WavFileBot, "on_call_started")
    assert hasattr(WavFileBot, "on_call_ended")


# ----------------------------------------------------------------------
# Логи и метрики задержек
# ----------------------------------------------------------------------
def test_setup_logging_writes_utf8_file(tmp_path):
    import logging

    log_path = tmp_path / "logs" / "call.log"
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level

    main.setup_logging(False, str(log_path))
    try:
        logging.getLogger("sipbot.test").info("Проверка UTF-8: задержка")
        for handler in root.handlers:
            handler.flush()
        text = log_path.read_text(encoding="utf-8")
    finally:
        for handler in list(root.handlers):
            if handler not in original_handlers:
                root.removeHandler(handler)
                handler.close()
        for handler in original_handlers:
            if handler not in root.handlers:
                root.addHandler(handler)
        root.setLevel(original_level)

    assert "Проверка UTF-8" in text


def test_log_latency_summary_writes_jsonl(tmp_path):
    from sipbot.metrics import LatencySample

    writer = main.MetricsWriter(tmp_path / "metrics.jsonl")
    bot = SimpleNamespace(
        latency_reports=[
            LatencySample(
                call_id="call-1",
                asr_ms=100.0,
                llm_ms=200.0,
                tts_ms=300.0,
                total_ms=600.0,
            )
        ]
    )

    main.log_latency_summary(bot, writer)

    lines = writer.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"call_id": "call-1"' in lines[0]
    assert '"total_ms": 600.0' in lines[0]


def test_log_latency_summary_skips_bots_without_metrics(tmp_path):
    writer = main.MetricsWriter(tmp_path / "metrics.jsonl")

    main.log_latency_summary(SimpleNamespace(), writer)

    assert not writer.path.exists()


def test_metrics_and_log_file_defaults_come_from_env(monkeypatch):
    monkeypatch.delenv("SIPBOT_LOG_FILE", raising=False)
    monkeypatch.delenv("SIPBOT_METRICS_FILE", raising=False)
    args = main.build_parser().parse_args(["dial", "+79780481885"])
    assert args.log_file is None and args.metrics_file is None
    assert main.metrics_writer_for(args) is None

    monkeypatch.setenv("SIPBOT_LOG_FILE", "call.log")
    monkeypatch.setenv("SIPBOT_METRICS_FILE", "metrics.jsonl")
    args = main.build_parser().parse_args(["dial", "+79780481885"])

    assert args.log_file == "call.log"
    assert args.metrics_file == "metrics.jsonl"
    assert main.metrics_writer_for(args).path.name == "metrics.jsonl"
