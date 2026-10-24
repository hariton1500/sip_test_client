"""Энергетический VAD для голосового бота.

Накапливает входящие u8-кадры, определяет начало речи по превышению порога
RMS и конец речи по тишине. Готовый WAV (s16le, 16 kГц, mono) возвращается
для отправки в ASR.
"""

from __future__ import annotations

import io
import wave
from typing import List

from .audio_utils import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, rms_u8, u8_to_s16


class EnergyVAD:
    """Простой детектор речи по энергии кадра.

    Args:
        speech_ms: минимальная длительность речи, чтобы считать реплику начавшейся.
        silence_ms: сколько мс тишины считать концом реплики.
        max_utterance_ms: жёсткий лимит длительности реплики.
        rms_threshold: порог RMS во внутреннем формате u8 (0..128).
    """

    def __init__(
        self,
        speech_ms: int = 300,
        silence_ms: int = 900,
        max_utterance_ms: int = 10000,
        rms_threshold: int = 12,
    ) -> None:
        self.rms_threshold = rms_threshold
        self.speech_frames = max(1, speech_ms // FRAME_MS)
        self.silence_frames = max(1, silence_ms // FRAME_MS)
        self.max_frames = max_utterance_ms // FRAME_MS

        self._buffer: List[bytes] = []
        self._speech_count = 0
        self._silence_count = 0
        self._has_speech = False
        self._closed = False

    def reset(self) -> None:
        """Очистить текущую реплику."""
        self._buffer.clear()
        self._speech_count = 0
        self._silence_count = 0
        self._has_speech = False

    def feed(self, frame: bytes) -> bool:
        """Добавить кадр. Возвращает True, если реплика готова к распознаванию."""
        if self._closed:
            return False
        if not frame:
            return False

        self._buffer.append(frame)
        level = rms_u8(frame)
        is_speech = level >= self.rms_threshold

        if is_speech:
            self._speech_count += 1
            self._silence_count = 0
            if self._speech_count >= self.speech_frames:
                self._has_speech = True
        else:
            self._silence_count += 1

        if self._has_speech:
            # конец реплики: достаточно тишины
            if self._silence_count >= self.silence_frames:
                return True
            # или превышен максимум длины
            if len(self._buffer) >= self.max_frames:
                return True
        return False

    def flush(self) -> bytes:
        """Собрать WAV s16le 16 kГц mono из накопленных u8-кадров и сбросить VAD."""
        pcm_u8 = b"".join(self._buffer)
        self.reset()
        return _pack_wav(u8_to_s16(pcm_u8), sample_rate=SAMPLE_RATE)

    def close(self) -> None:
        self._closed = True
        self.reset()


def _pack_wav(pcm_s16: bytes, sample_rate: int) -> bytes:
    """Упаковать s16le PCM в WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_s16)
    return buf.getvalue()


__all__ = ["EnergyVAD"]
