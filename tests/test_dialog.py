"""Тесты диалогового бота: ASR → LLM → TTS, приветствие и barge-in."""

from __future__ import annotations

import time

from sipbot.ai_client import ChatResponse
from sipbot.audio_utils import SILENCE_FRAME, tone_u8
from sipbot.dialog import DialogBot, DialogConfig
from sipbot.transcript import BOT, CALLER, SYSTEM, TranscriptStore
from sipbot.vad import EnergyVAD


class FakeTts:
    def __init__(self, audio: bytes | None = None) -> None:
        self.audio = audio or tone_u8(440, 100, amplitude=0.4)
        self.calls = []

    def preload(self, text: str) -> bytes:
        self.calls.append(text)
        return self.audio

    def audio_u8(self, text: str) -> bytes:
        self.calls.append(text)
        return self.audio


class FakeAi:
    def __init__(self, reply: str = "Ваш баланс 100 рублей.") -> None:
        self.reply = reply
        self.transcribe_calls = []
        self.chat_calls = []

    def transcribe(self, wav_bytes, language=None, prompt=None):
        self.transcribe_calls.append((wav_bytes, language, prompt))
        return "какой у меня баланс"

    def chat(self, messages):
        self.chat_calls.append(messages)
        return ChatResponse(content=self.reply)


def _config(**overrides) -> DialogConfig:
    values = dict(
        greeting="",
        fallback_reply="Повторите, пожалуйста.",
        history_turns=4,
        max_reply_chars=300,
        barge_in=True,
        vad_speech_ms=20,
        vad_silence_ms=20,
        vad_max_utterance_ms=500,
        tts_cache_dir=None,
    )
    values.update(overrides)
    return DialogConfig(**values)


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_dialog_runs_asr_llm_tts_and_records_latency():
    store = TranscriptStore()
    ai = FakeAi()
    tts = FakeTts()
    bot = DialogBot(
        ai,
        "Ты оператор поддержки.",
        store,
        config=_config(),
        vad=EnergyVAD(speech_ms=20, silence_ms=20, max_utterance_ms=500),
        tts=tts,
    )

    bot.on_call_started("+79780481885", "call-1")
    speech = tone_u8(440, 20, amplitude=0.5)
    bot.on_frame(speech)
    bot.on_frame(SILENCE_FRAME)

    frames = []

    def has_reply():
        frames.append(bot.on_frame(SILENCE_FRAME))
        return bool(bot.latency_reports)

    assert _wait_for(has_reply)
    # Забираем оставшиеся кадры TTS из внутренней очереди бота.
    for _ in range(20):
        frame = bot.on_frame(SILENCE_FRAME)
        frames.append(frame)
    bot.close()

    messages = store.snapshot()
    assert [m["text"] for m in messages if m["role"] == CALLER] == [
        "какой у меня баланс"
    ]
    assert [m["text"] for m in messages if m["role"] == BOT] == [
        "Ваш баланс 100 рублей."
    ]
    assert ai.chat_calls[0][0] == {
        "role": "system",
        "content": "Ты оператор поддержки.",
    }
    assert ai.chat_calls[0][1] == {
        "role": "user",
        "content": "какой у меня баланс",
    }
    assert any(frame and frame != SILENCE_FRAME for frame in frames)
    assert len(bot.latency_reports) == 1
    report = bot.last_latency
    assert report is not None
    assert report.call_id == "call-1"
    assert report.asr_ms >= 0 and report.llm_ms >= 0 and report.tts_ms >= 0
    assert report.total_ms >= report.asr_ms + report.llm_ms + report.tts_ms


def test_dialog_queues_full_greeting_and_barge_in_clears_it():
    store = TranscriptStore()
    ai = FakeAi()
    greeting_audio = tone_u8(440, 6000, amplitude=0.3)  # 6 секунд = 300 кадров
    bot = DialogBot(
        ai,
        "system",
        store,
        config=_config(greeting="Здравствуйте!", barge_in=True),
        vad=EnergyVAD(speech_ms=20, silence_ms=20, max_utterance_ms=500),
        tts=FakeTts(greeting_audio),
        greeting_audio=greeting_audio,
    )

    bot.on_call_started("caller", "call-2")
    assert bot.pending_output_frames == 300

    bot.on_frame(tone_u8(440, 20, amplitude=0.5))
    assert bot.pending_output_frames == 0
    assert any("перебил" in m["text"] for m in store.snapshot())
    bot.close()


def test_dialog_close_transcribes_final_utterance_after_hangup():
    store = TranscriptStore()
    ai = FakeAi()
    bot = DialogBot(
        ai,
        "system",
        store,
        config=_config(),
        vad=EnergyVAD(speech_ms=20, silence_ms=20, max_utterance_ms=500),
        tts=FakeTts(),
    )

    bot.on_call_started("caller", "call-3")
    bot.on_frame(tone_u8(440, 20, amplitude=0.5))
    bot.on_call_ended()
    bot.close(grace=1.5)

    assert [m["text"] for m in store.snapshot() if m["role"] == CALLER] == [
        "какой у меня баланс"
    ]
    assert ai.chat_calls == []


def test_dialog_config_reads_env():
    config = DialogConfig.from_env(
        {
            "AI_GREETING": "  Добрый день!  ",
            "DIALOG_HISTORY_TURNS": "5",
            "DIALOG_MAX_REPLY_CHARS": "500",
            "DIALOG_BARGE_IN": "off",
            "VAD_SPEECH_MS": "250",
            "VAD_SILENCE_MS": "700",
            "VAD_MAX_UTTERANCE_MS": "8000",
            "VAD_RMS_THRESHOLD": "0.03",
            "TTS_CACHE_DIR": "",
        }
    )

    assert config.greeting == "Добрый день!"
    assert config.history_turns == 5
    assert config.max_reply_chars == 500
    assert config.barge_in is False
    assert config.vad_speech_ms == 250
    assert config.vad_silence_ms == 700
    assert config.vad_max_utterance_ms == 8000
    assert config.vad_rms_threshold == 0.03
    assert config.tts_cache_dir is None
