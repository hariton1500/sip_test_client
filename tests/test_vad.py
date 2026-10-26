"""Тесты EnergyVAD: тишина не должна попадать в ASR и съедать лимит реплики."""

from __future__ import annotations

import io
import wave

from sipbot.audio_utils import FRAME_MS, SILENCE_FRAME, tone_u8
from sipbot.vad import EnergyVAD


def _tone_frame() -> bytes:
    return tone_u8(440, FRAME_MS, amplitude=0.5)


def _wav_ms(wav_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return round(wf.getnframes() * 1000 / wf.getframerate())


def test_long_silence_is_not_buffered_before_speech():
    vad = EnergyVAD(
        speech_ms=40,
        silence_ms=100,
        max_utterance_ms=5000,
        pre_roll_ms=200,
    )

    for _ in range(1500):  # 30 секунд тишины
        assert vad.feed(SILENCE_FRAME) is False
    assert vad.has_speech is False

    for _ in range(50):  # 1 секунда речи
        vad.feed(_tone_frame())

    ready = False
    for _ in range(5):  # 100 мс тишины
        ready = vad.feed(SILENCE_FRAME)
    assert ready is True

    wav = vad.flush()
    info = vad.last_flush_info
    assert info is not None

    # 200 мс pre-roll + 1 с речи + 100 мс хвоста, а не 30+ секунд.
    assert 1000 <= _wav_ms(wav) <= 1400
    assert info.speech_ms == 1000
    assert info.buffered_ms == _wav_ms(wav)
    assert info.trailing_silence_ms == 100
    assert info.reason == "silence"


def test_max_utterance_counts_speech_not_buffered_silence():
    vad = EnergyVAD(
        speech_ms=40,
        silence_ms=2000,
        max_utterance_ms=500,
        pre_roll_ms=200,
    )

    for _ in range(250):  # 5 секунд тишины не должны приближать лимит
        assert vad.feed(SILENCE_FRAME) is False

    ready_at = None
    for index in range(200):
        if vad.feed(_tone_frame()):
            ready_at = index + 1
            break

    assert ready_at == 25  # лимит 500 мс считается по 20-мс кадрам речи
    wav = vad.flush()
    info = vad.last_flush_info
    assert info is not None
    assert info.reason == "max"
    assert info.speech_ms == 500
    assert info.buffered_ms == 660  # 200 мс pre-roll + 460 мс после старта
    assert _wav_ms(wav) == 660


def test_buffered_length_does_not_trigger_max_until_enough_speech():
    vad = EnergyVAD(
        speech_ms=40,
        silence_ms=1000,
        max_utterance_ms=500,
        pre_roll_ms=200,
    )

    vad.feed(_tone_frame())
    vad.feed(_tone_frame())  # речь началась: 40 мс
    vad.feed(SILENCE_FRAME)
    for _ in range(22):  # всего 480 мс речи, но буфер уже 660 мс
        assert vad.feed(_tone_frame()) is False

    assert vad.feed(_tone_frame()) is True  # ровно 500 мс речи
    vad.flush()
    info = vad.last_flush_info
    assert info is not None
    assert info.reason == "max"
    assert info.speech_ms == 500


def test_idle_silence_keeps_only_short_pre_roll():
    vad = EnergyVAD(speech_ms=100, silence_ms=500, pre_roll_ms=100)

    for _ in range(3000):  # минута тишины
        assert vad.feed(SILENCE_FRAME) is False

    assert vad._buffer == []  # noqa: SLF001 - проверяем отсутствие утечки
    assert len(vad._pre_roll) == 5  # noqa: SLF001 - только 100 мс pre-roll


def test_short_blip_does_not_start_utterance():
    vad = EnergyVAD(speech_ms=100, silence_ms=500, pre_roll_ms=200)

    for _ in range(4):  # 80 мс — меньше порога 100 мс
        assert vad.feed(_tone_frame()) is False
    vad.feed(SILENCE_FRAME)
    for _ in range(4):
        assert vad.feed(_tone_frame()) is False

    assert vad.has_speech is False
    assert vad._speech_count == 4  # noqa: SLF001 - счётчик сбрасывался тишиной
