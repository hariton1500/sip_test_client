"""Тесты ленты разговора, ASR-обёртки и веб-интерфейса (без сети)."""

from __future__ import annotations

import json
import time
import urllib.request

from sipbot.audio_utils import SILENCE_FRAME, tone_u8
from sipbot.bots import EchoBot
from sipbot.transcript import CALLER, SYSTEM, TranscriptStore
from sipbot.transcribing_bot import TranscribingBot
from sipbot.ui_server import UiServer
from sipbot.vad import EnergyVAD


class FakeAi:
    def __init__(self, text: str = "проверка связи") -> None:
        self.text = text
        self.calls = []

    def transcribe(self, wav_bytes, language=None, prompt=None):
        self.calls.append((wav_bytes, language, prompt))
        return self.text


def test_store_publishes_and_notifies_listener():
    store = TranscriptStore()
    listener = store.subscribe()

    payload = store.publish(CALLER, "  привет   мир ", call_id="call-1")

    assert payload is not None
    assert payload["text"] == "привет мир"
    event = listener.get_nowait()
    assert event["type"] == "message"
    assert event["role"] == CALLER
    assert store.snapshot()[0]["call_id"] == "call-1"

    store.unsubscribe(listener)
    assert store.listener_count == 0


def test_store_ignores_empty_and_normalizes_role():
    store = TranscriptStore()
    assert store.publish(CALLER, "   ") is None
    payload = store.publish("unknown", "system text")
    assert payload["role"] == SYSTEM


def test_store_clear_notifies_listeners():
    store = TranscriptStore()
    store.publish(CALLER, "одна реплика")
    listener = store.subscribe()

    store.clear()

    assert listener.get_nowait() == {"type": "clear"}
    assert store.snapshot() == []


def test_transcribing_bot_publishes_asr_text():
    store = TranscriptStore()
    ai = FakeAi()
    vad = EnergyVAD(speech_ms=20, silence_ms=20, max_utterance_ms=500)
    bot = TranscribingBot(EchoBot(), store, ai, vad=vad, language="ru")

    bot.on_call_started("1001", "call-42")
    speech = tone_u8(440, 20, amplitude=0.5)
    assert bot.on_frame(speech) == speech
    bot.on_frame(SILENCE_FRAME)

    deadline = time.monotonic() + 2.0
    messages = []
    while time.monotonic() < deadline:
        messages = [m for m in store.snapshot() if m["role"] == CALLER]
        if messages:
            break
        time.sleep(0.02)

    bot.close()

    assert [m["text"] for m in messages] == ["проверка связи"]
    assert messages[0]["call_id"] == "call-42"
    assert ai.calls and ai.calls[0][1] == "ru"
    assert any(m["role"] == SYSTEM for m in store.snapshot())


def test_transcribing_bot_close_processes_final_utterance():
    store = TranscriptStore()
    ai = FakeAi("последняя фраза")
    vad = EnergyVAD(speech_ms=20, silence_ms=20, max_utterance_ms=500)
    bot = TranscribingBot(EchoBot(), store, ai, vad=vad, language="ru")
    bot.on_call_started("1001", "call-fast-hangup")
    bot.on_frame(tone_u8(440, 20, amplitude=0.5))
    bot.on_call_ended()
    bot.close(grace=1.0)

    assert [m["text"] for m in store.snapshot() if m["role"] == CALLER] == [
        "последняя фраза"
    ]


def test_transcribing_bot_without_ai_only_tracks_events():
    store = TranscriptStore()
    bot = TranscribingBot(None, store, None)
    bot.on_call_started("1002", "call-7")
    bot.on_frame(tone_u8(440, 20, amplitude=0.5))
    bot.on_call_ended()
    bot.close()

    roles = [m["role"] for m in store.snapshot()]
    assert roles == [SYSTEM, SYSTEM]


def test_ui_server_streams_sse_snapshot_and_updates():
    store = TranscriptStore()
    store.publish(CALLER, "старая реплика", call_id="sse-call")
    server = UiServer(store, host="127.0.0.1", port=0)
    server.start()
    try:
        url = f"http://127.0.0.1:{server.port}/events"
        with urllib.request.urlopen(url, timeout=3) as response:
            assert response.readline().decode().startswith("event: snapshot")
            data = json.loads(response.readline().decode().removeprefix("data: "))
            assert data["messages"][0]["text"] == "старая реплика"
            assert response.readline() == b"\n"

            store.publish(CALLER, "новая реплика", call_id="sse-call")
            assert response.readline().decode().startswith("event: message")
            update = json.loads(response.readline().decode().removeprefix("data: "))
            assert update["text"] == "новая реплика"
    finally:
        server.stop()


def test_ui_server_serves_page_and_messages():
    store = TranscriptStore()
    store.publish(CALLER, "тестовая реплика", call_id="ui-call")
    server = UiServer(store, host="127.0.0.1", port=0)
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        with urllib.request.urlopen(base + "/", timeout=2) as response:
            html = response.read().decode("utf-8")
        assert "Разговор в реальном времени" in html

        with urllib.request.urlopen(base + "/api/messages", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["messages"][0]["text"] == "тестовая реплика"

        request = urllib.request.Request(base + "/api/clear", method="POST")
        with urllib.request.urlopen(request, timeout=2) as response:
            assert response.status == 200
        assert store.snapshot() == []
    finally:
        server.stop()
