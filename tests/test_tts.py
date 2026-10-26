"""Тесты TTS-кэша и приведения WAV к формату pyVoIP."""

from __future__ import annotations

import io
import wave

import pytest

from sipbot import audio_utils as au
from sipbot.audio_utils import tone_s16
from sipbot.tts import TtsService, wav_bytes_to_pcm


def _wav_bytes(pcm_s16: bytes, rate: int, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm_s16)
    return buffer.getvalue()


class FakeAi:
    tts_model = "test-tts"
    tts_voice = "test-voice"

    def __init__(self, wav: bytes) -> None:
        self.wav = wav
        self.calls = []

    def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        return self.wav


def test_tts_resamples_to_8khz_and_uses_cache(tmp_path):
    wav = _wav_bytes(tone_s16(440, 100, rate=16000), 16000)
    ai = FakeAi(wav)
    cache_dir = tmp_path / "tts"

    service = TtsService(ai, cache_dir=cache_dir)
    first = service.audio_u8("  привет   мир ")
    second = service.audio_u8("привет мир")

    assert len(first) == 800  # 100 мс u8 при 8 кГц
    assert first == second
    assert ai.calls == ["привет мир"]
    assert len(list(cache_dir.glob("*.wav"))) == 1

    # Новый экземпляр сервиса читает готовый WAV с диска.
    again = TtsService(ai, cache_dir=cache_dir).audio_u8("привет мир")
    assert again == first
    assert ai.calls == ["привет мир"]


def test_tts_accepts_raw_pcm_and_caches_wav(tmp_path):
    raw_pcm = tone_s16(440, 100, rate=24000)

    class RawPcmAi(FakeAi):
        tts_sample_rate = 24000

    ai = RawPcmAi(raw_pcm)
    cache_dir = tmp_path / "tts"
    audio = TtsService(ai, cache_dir=cache_dir).audio_u8("привет")

    assert len(audio) == 800
    cached = next(cache_dir.glob("*.wav"))
    pcm, rate = wav_bytes_to_pcm(cached.read_bytes())
    assert rate == 24000
    assert pcm == raw_pcm


def test_tts_trims_long_edge_silence():
    """Gemini добавляет ~0.4 с тишины на края каждого предложения."""
    pcm = au.silence_s16(400) + tone_s16(440, 200) + au.silence_s16(400)
    ai = FakeAi(_wav_bytes(pcm, 8000))
    service = TtsService(ai, cache_dir=None)

    audio = service.audio_u8("привет")

    # 200 мс речи + по 40 мс запаса с каждой стороны (8 кГц, 1 байт = 1 сэмпл).
    assert 2200 <= len(audio) <= 2400


def test_tts_keeps_short_fragments_intact():
    pcm = tone_s16(440, 20, rate=8000)
    service = TtsService(FakeAi(_wav_bytes(pcm, 8000)), cache_dir=None)

    assert len(service.audio_u8("да")) == 160


def test_tts_survives_unusable_cache_dir(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x", encoding="utf-8")
    ai = FakeAi(_wav_bytes(tone_s16(440, 20, rate=8000), 8000))

    service = TtsService(ai, cache_dir=blocker / "tts")

    assert service.cache_dir is None
    assert len(service.audio_u8("привет")) == 160  # 20 мс u8 при 8 кГц
    assert ai.calls == ["привет"]


def test_tts_empty_text_does_not_call_api(tmp_path):
    ai = FakeAi(_wav_bytes(tone_s16(440, 20, rate=8000), 8000))
    service = TtsService(ai, cache_dir=tmp_path / "tts")

    assert service.audio_u8("   ") == b""
    assert ai.calls == []


def test_wav_bytes_to_pcm_downmixes_stereo():
    left = tone_s16(440, 20, rate=8000)
    right = tone_s16(440, 20, rate=8000)
    interleaved = b"".join(
        left[index : index + 2] + right[index : index + 2]
        for index in range(0, len(left), 2)
    )

    pcm, rate = wav_bytes_to_pcm(_wav_bytes(interleaved, 8000, channels=2))

    assert rate == 8000
    assert pcm == left


def test_wav_bytes_to_pcm_rejects_non_wav():
    with pytest.raises(ValueError, match="не WAV"):
        wav_bytes_to_pcm(b"definitely not a wave file")
