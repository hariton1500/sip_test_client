"""Тесты диалогового бота: ASR → LLM → TTS, приветствие и barge-in."""

from __future__ import annotations

import json
import threading
import time

from sipbot.ai_client import ChatResponse
from sipbot.audio_utils import SILENCE_FRAME, tone_u8
from sipbot.dialog import DialogBot, DialogConfig, _is_asr_hallucination
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
    def __init__(
        self,
        reply: str = "Ваш баланс 100 рублей.",
        transcription: str = "какой у меня баланс",
    ) -> None:
        self.reply = reply
        self.transcription = transcription
        self.transcribe_calls = []
        self.chat_calls = []

    def transcribe(self, wav_bytes, language=None, prompt=None):
        self.transcribe_calls.append((wav_bytes, language, prompt))
        return self.transcription

    def chat(self, messages, tools=None):
        self.chat_calls.append(messages)
        return ChatResponse(content=self.reply)


class SequencedAi(FakeAi):
    """Медленный первый ASR, чтобы вторая реплика успела встать в очередь."""

    def __init__(self, transcriptions, replies) -> None:
        super().__init__()
        self.transcriptions = list(transcriptions)
        self.replies = list(replies)
        self.first_started = threading.Event()
        self.release_first = threading.Event()

    def transcribe(self, wav_bytes, language=None, prompt=None):
        index = len(self.transcribe_calls)
        self.transcribe_calls.append((wav_bytes, language, prompt))
        if index == 0:
            self.first_started.set()
            assert self.release_first.wait(2.0)
        return self.transcriptions[index]

    def chat(self, messages, tools=None):
        index = len(self.chat_calls)
        self.chat_calls.append(messages)
        return ChatResponse(content=self.replies[index])


class FakeBilling:
    def __init__(self) -> None:
        self.accounts = []

    def get_account(self, account_id: str):
        self.accounts.append(account_id)
        return {
            "found": True,
            "user": {"id": account_id, "extra_account": 429.56},
        }


class ToolAi:
    """Первый ответ — tool_call, второй — финальный текст."""

    def __init__(self, tool_name: str, arguments: str, final: str) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self.final = final
        self.chat_calls = []
        self.transcribe_calls = []

    def transcribe(self, wav_bytes, language=None, prompt=None):
        self.transcribe_calls.append((wav_bytes, language, prompt))
        return "какой у меня баланс"

    def chat(self, messages, tools=None):
        self.chat_calls.append((messages, tools))
        if len(self.chat_calls) == 1:
            return ChatResponse(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": self.tool_name,
                            "arguments": self.arguments,
                        },
                    }
                ],
            )
        return ChatResponse(content=self.final)


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
    assert report.queue_ms >= 0
    assert report.total_ms >= (
        report.queue_ms + report.asr_ms + report.llm_ms + report.tts_ms
    )


def test_dialog_calls_billing_tool_and_passes_result_to_llm():
    store = TranscriptStore()
    billing = FakeBilling()
    ai = ToolAi(
        "get_billing_info",
        '{"account_id":"20"}',
        "На вашем счёте четыреста двадцать девять рублей пятьдесят шесть копеек.",
    )
    bot = DialogBot(
        ai,
        "system",
        store,
        config=_config(),
        vad=EnergyVAD(speech_ms=20, silence_ms=20, max_utterance_ms=500),
        tts=FakeTts(),
        billing=billing,
    )

    bot.on_call_started("caller", "call-tool")
    bot.on_frame(tone_u8(440, 20, amplitude=0.5))
    bot.on_frame(SILENCE_FRAME)
    assert _wait_for(lambda: len(bot.latency_reports) == 1)
    bot.close()

    assert billing.accounts == ["20"]
    assert len(ai.chat_calls) == 2
    first_messages, first_tools = ai.chat_calls[0]
    tool_names = {
        item["function"]["name"] for item in (first_tools or [])
    }
    assert {"get_billing_info", "end_call"} <= tool_names
    user_message = next(m for m in first_messages if m["role"] == "user")
    assert user_message["content"] == "какой у меня баланс"

    second_messages = ai.chat_calls[1][0]
    tool_message = [m for m in second_messages if m["role"] == "tool"][-1]
    assert tool_message["tool_call_id"] == "call-1"
    result = json.loads(tool_message["content"])
    assert result["user"]["extra_account"] == 429.56
    assert bot.last_latency.text.startswith("На вашем счёте")


def test_dialog_records_llm_rounds_and_tts_chunks():
    """Детализация: раунды LLM (tool-цикл) и предложения TTS попадают в отчёт."""
    ai = ToolAi(
        "get_billing_info",
        '{"account_id":"20"}',
        "Первое предложение ответа. Второе предложение ответа.",
    )
    bot = DialogBot(
        ai,
        "system",
        config=_config(),
        vad=EnergyVAD(speech_ms=20, silence_ms=20, max_utterance_ms=500),
        tts=FakeTts(),
        billing=FakeBilling(),
    )

    bot.on_call_started("caller", "call-details")
    bot.on_frame(tone_u8(440, 20, amplitude=0.5))
    bot.on_frame(SILENCE_FRAME)
    assert _wait_for(lambda: len(bot.latency_reports) == 1)
    bot.close()

    report = bot.last_latency
    assert report is not None
    assert len(report.llm_rounds_ms) == 2
    assert all(value >= 0 for value in report.llm_rounds_ms)
    assert report.tool_ms >= 0
    assert len(report.tts_chunks_ms) == 2
    assert report.tts_first_chunk_ms == report.tts_chunks_ms[0]
    assert report.tts_ms >= sum(report.tts_chunks_ms)


def test_dialog_end_call_reports_ready_only_after_audio_played():
    ai = ToolAi("end_call", "{}", "Всего доброго!")
    bot = DialogBot(
        ai,
        "system",
        config=_config(),
        vad=EnergyVAD(speech_ms=20, silence_ms=20, max_utterance_ms=500),
        tts=FakeTts(),
    )

    bot.on_call_started("caller", "call-end")
    bot.on_frame(tone_u8(440, 20, amplitude=0.5))
    bot.on_frame(SILENCE_FRAME)
    assert _wait_for(lambda: len(bot.latency_reports) == 1)

    # Пока TTS-кадры не разобраны, трубку вешать нельзя.
    assert bot.pending_output_frames > 0
    assert bot.end_call_requested is False
    for _ in range(100):
        if bot.pending_output_frames == 0:
            break
        bot.on_frame(SILENCE_FRAME)

    assert bot.pending_output_frames == 0
    assert bot.end_call_requested is True
    bot.close()


def test_dialog_barge_in_cancels_pending_end_call():
    ai = ToolAi("end_call", "{}", "Всего доброго!")
    bot = DialogBot(
        ai,
        "system",
        config=_config(),
        vad=EnergyVAD(speech_ms=20, silence_ms=20, max_utterance_ms=500),
        tts=FakeTts(),
    )

    bot.on_call_started("caller", "call-end-barge-in")
    bot.on_frame(tone_u8(440, 20, amplitude=0.5))
    bot.on_frame(SILENCE_FRAME)
    assert _wait_for(lambda: len(bot.latency_reports) == 1)
    assert bot.pending_output_frames > 0

    # Абонент перебил прощание — вешать трубку нельзя.
    bot.on_frame(tone_u8(440, 20, amplitude=0.5))
    assert bot.pending_output_frames == 0
    assert bot.end_call_requested is False
    bot.close()


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


def test_dialog_new_speech_while_preparing_cancels_stale_reply_without_barge_in():
    store = TranscriptStore()
    ai = SequencedAi(
        transcriptions=["первый вопрос", "второй вопрос"],
        replies=["новый ответ"],
    )
    bot = DialogBot(
        ai,
        "system",
        store,
        config=_config(),
        vad=EnergyVAD(speech_ms=20, silence_ms=20, max_utterance_ms=500),
        tts=FakeTts(),
    )

    bot.on_call_started("caller", "call-4")
    speech = tone_u8(440, 20, amplitude=0.5)

    # Первая реплика ушла в ASR и «зависла».
    bot.on_frame(speech)
    bot.on_frame(SILENCE_FRAME)
    assert ai.first_started.wait(1.0)

    # Абонент начал говорить снова, пока аудио ещё не играет: это не barge-in,
    # но старый ответ после подготовки отдавать нельзя.
    bot.on_frame(speech)
    bot.on_frame(SILENCE_FRAME)
    assert not any("перебил" in m["text"] for m in store.snapshot())
    assert any("уточнён" in m["text"] for m in store.snapshot())

    ai.release_first.set()
    assert _wait_for(lambda: len(bot.latency_reports) == 1)
    bot.close()

    assert [m["text"] for m in store.snapshot() if m["role"] == CALLER] == [
        "первый вопрос",
        "второй вопрос",
    ]
    assert [m["text"] for m in store.snapshot() if m["role"] == BOT] == [
        "новый ответ"
    ]
    # LLM вызван один раз — для второй реплики; история содержит оба вопроса.
    assert len(ai.chat_calls) == 1
    assert [m["content"] for m in ai.chat_calls[0][1:]] == [
        "первый вопрос",
        "второй вопрос",
    ]
    report = bot.last_latency
    assert report is not None
    assert report.text == "новый ответ"
    assert report.queue_ms > 0


def test_dialog_drops_whisper_hallucination_without_llm():
    store = TranscriptStore()
    ai = FakeAi(transcription="Продолжение следует... Продолжение следует...")
    bot = DialogBot(
        ai,
        "system",
        store,
        config=_config(),
        vad=EnergyVAD(speech_ms=20, silence_ms=20, max_utterance_ms=500),
        tts=FakeTts(),
    )

    bot.on_call_started("caller", "call-5")
    bot.on_frame(tone_u8(440, 20, amplitude=0.5))
    bot.on_frame(SILENCE_FRAME)

    assert _wait_for(lambda: len(ai.transcribe_calls) == 1)
    time.sleep(0.05)
    bot.close()

    assert [m for m in store.snapshot() if m["role"] == CALLER] == []
    assert ai.chat_calls == []
    assert bot.latency_reports == []


def test_asr_hallucination_filter_keeps_short_real_answers():
    assert _is_asr_hallucination("Продолжение следует... Продолжение следует...")
    assert _is_asr_hallucination("Субтитры сделал DimaTorzok")
    assert _is_asr_hallucination("Спасибо за просмотр!")
    assert not _is_asr_hallucination("Спасибо.")
    assert not _is_asr_hallucination("Да.")
    assert not _is_asr_hallucination("Спасибо, всё понятно.")


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


def test_dialog_reply_limited_to_two_sentences():
    bot = DialogBot(
        FakeAi(),
        "system",
        config=_config(max_reply_sentences=2),
    )
    assert bot._prepare_reply("Раз. Два! Три?") == "Раз. Два!"

    one = DialogBot(
        FakeAi(),
        "system",
        config=_config(max_reply_sentences=1),
    )
    assert one._prepare_reply("Раз. Два!") == "Раз."


def test_dialog_streams_tts_by_sentence_and_reports_first_audio():
    ai = FakeAi(reply="Первое предложение. Второе предложение.")
    tts = FakeTts()
    bot = DialogBot(
        ai,
        "system",
        config=_config(),
        vad=EnergyVAD(speech_ms=20, silence_ms=20, max_utterance_ms=500),
        tts=tts,
    )

    bot.on_call_started("caller", "call-stream")
    bot.on_frame(tone_u8(440, 20, amplitude=0.5))
    bot.on_frame(SILENCE_FRAME)
    assert _wait_for(lambda: len(bot.latency_reports) == 1)
    bot.close()

    assert tts.calls == ["Первое предложение.", "Второе предложение."]
    report = bot.last_latency
    assert report is not None
    assert 0 < report.first_audio_ms <= report.total_ms


def test_dialog_reply_chunks_keeps_short_sentences_together():
    assert DialogBot._reply_chunks("Да. Сейчас проверю баланс.") == [
        "Да. Сейчас проверю баланс."
    ]
    assert DialogBot._reply_chunks("Первое. Второе.") == [
        "Первое. Второе."
    ]


def test_dialog_no_input_prompt_then_auto_hangup():
    config = _config(no_input_prompt_ms=10, no_input_hangup_ms=1000)
    store = TranscriptStore()
    bot = DialogBot(
        FakeAi(),
        "system",
        store,
        config=config,
        vad=EnergyVAD(speech_ms=20, silence_ms=20, max_utterance_ms=500),
        tts=FakeTts(),
    )

    bot.on_call_started("caller", "call-idle")
    time.sleep(0.03)
    bot.on_frame(SILENCE_FRAME)
    assert _wait_for(
        lambda: any(
            m["role"] == BOT and m["text"] == config.no_input_prompt
            for m in store.snapshot()
        )
    )

    for _ in range(100):
        if bot.pending_output_frames == 0:
            break
        bot.on_frame(SILENCE_FRAME)

    # Ещё один кадр фиксирует «бот договорил», после чего имитируем
    # истёкшие 60 секунд тишины.
    bot.on_frame(SILENCE_FRAME)
    bot._quiet_since = time.monotonic() - 2.0
    bot._check_no_input(time.monotonic())
    assert bot.end_call_requested is True
    bot.close()


def test_dialog_config_defaults_use_low_latency_vad():
    config = DialogConfig.from_env({})

    assert config.vad_silence_ms == 600
    assert config.vad_pre_roll_ms == 200


def test_dialog_config_reads_env():
    config = DialogConfig.from_env(
        {
            "AI_GREETING": "  Добрый день!  ",
            "DIALOG_HISTORY_TURNS": "5",
            "DIALOG_MAX_REPLY_CHARS": "500",
            "DIALOG_MAX_REPLY_SENTENCES": "3",
            "DIALOG_NO_INPUT_PROMPT_MS": "15000",
            "DIALOG_NO_INPUT_HANGUP_MS": "90000",
            "DIALOG_NO_INPUT_PROMPT": "Вы здесь?",
            "DIALOG_BARGE_IN": "off",
            "VAD_SPEECH_MS": "250",
            "VAD_SILENCE_MS": "700",
            "VAD_MAX_UTTERANCE_MS": "8000",
            "VAD_RMS_THRESHOLD": "0.03",
            "VAD_PREROLL_MS": "250",
            "TTS_CACHE_DIR": "",
        }
    )

    assert config.greeting == "Добрый день!"
    assert config.history_turns == 5
    assert config.max_reply_chars == 500
    assert config.max_reply_sentences == 3
    assert config.no_input_prompt_ms == 15000
    assert config.no_input_hangup_ms == 90000
    assert config.no_input_prompt == "Вы здесь?"
    assert config.barge_in is False
    assert config.vad_speech_ms == 250
    assert config.vad_silence_ms == 700
    assert config.vad_max_utterance_ms == 8000
    assert config.vad_rms_threshold == 0.03
    assert config.vad_pre_roll_ms == 250
    assert config.tts_cache_dir is None
