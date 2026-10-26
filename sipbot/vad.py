"""Энергетический VAD для голосового бота.

Накапливает входящие u8-кадры, определяет начало речи по превышению порога
RMS и конец речи по тишине. Готовый WAV (s16le, 8 kГц, mono) возвращается
для отправки в ASR.
"""

from __future__ import annotations

import io
import wave
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

from .audio_utils import FRAME_MS, SAMPLE_RATE, rms_u8, u8_to_s16


@dataclass(frozen=True)
class VadFlushInfo:
    """Параметры только что отданной реплики.

    ``speech_ms`` — суммарная длительность кадров выше порога, ``buffered_ms`` —
    длина WAV с pre-roll и хвостовой тишиной, ``trailing_silence_ms`` — сколько
    тишины реально попало в конец (нужно для честного ``speech_end``).
    """

    speech_ms: int
    buffered_ms: int
    trailing_silence_ms: int
    reason: str


class EnergyVAD:
    """Простой детектор речи по энергии кадра.

    В режиме ожидания хранится только короткий pre-roll, а не вся тишина с
    момента прошлой реплики. Поэтому длительное молчание абонента не приводит
    к отправке в ASR многодесятисекундных WAV и не «съедает» лимит
    ``max_utterance_ms`` до начала речи.

    Args:
        speech_ms: минимальная длительность речи, чтобы считать реплику начавшейся.
        silence_ms: сколько мс тишины считать концом реплики.
        max_utterance_ms: жёсткий лимит длительности **речи** в реплике.
        rms_threshold: порог RMS (0.0..1.0), выше которого кадр считается речью.
        pre_roll_ms: сколько аудио до начала речи сохранить в начале WAV.
    """

    def __init__(
        self,
        speech_ms: int = 300,
        silence_ms: int = 900,
        max_utterance_ms: int = 10000,
        rms_threshold: float = 0.02,
        pre_roll_ms: int = 300,
    ) -> None:
        self.rms_threshold = rms_threshold
        self.speech_frames = max(1, speech_ms // FRAME_MS)
        self.silence_frames = max(1, silence_ms // FRAME_MS)
        self.max_frames = max(1, max_utterance_ms // FRAME_MS)
        self.pre_roll_frames = max(1, pre_roll_ms // FRAME_MS)

        self._pre_roll: Deque[bytes] = deque(maxlen=self.pre_roll_frames)
        self._buffer: List[bytes] = []
        self._speech_count = 0
        # Суммарная речь в реплике, а не длина буфера: лимит не должен
        # срабатывать из-за тишины перед фразой.
        self._utterance_speech_frames = 0
        self._silence_count = 0
        self._has_speech = False
        self._closed = False
        self._flush_reason = "manual"
        self._last_flush_info: Optional[VadFlushInfo] = None

    def reset(self) -> None:
        """Очистить текущую реплику."""
        self._pre_roll.clear()
        self._buffer.clear()
        self._speech_count = 0
        self._utterance_speech_frames = 0
        self._silence_count = 0
        self._has_speech = False
        self._flush_reason = "manual"
        self._last_flush_info = None

    @property
    def has_speech(self) -> bool:
        """Была ли в текущем буфере зафиксирована речь."""
        return self._has_speech

    @property
    def speech_active(self) -> bool:
        """Идёт ли сейчас непрерывный участок речи (для barge-in)."""
        return self._speech_count >= self.speech_frames

    @property
    def silence_ms(self) -> int:
        """Сколько тишины VAD требует для завершения реплики."""
        return self.silence_frames * FRAME_MS

    @property
    def pre_roll_ms(self) -> int:
        """Сколько аудио до речи сохраняется в начале реплики."""
        return self.pre_roll_frames * FRAME_MS

    @property
    def last_flush_info(self) -> Optional[VadFlushInfo]:
        """Информация о последней отданной реплике."""
        return self._last_flush_info

    def feed(self, frame: bytes) -> bool:
        """Добавить кадр. Возвращает True, если реплика готова к распознаванию."""
        if self._closed:
            return False
        if not frame:
            return False

        is_speech = rms_u8(frame) >= self.rms_threshold

        if not self._has_speech:
            # До начала речи храним только короткий pre-roll. Тишина не
            # накапливается и не влияет на max_utterance_ms.
            self._pre_roll.append(frame)
            if is_speech:
                self._speech_count += 1
            else:
                self._speech_count = 0

            if self._speech_count < self.speech_frames:
                return False

            self._has_speech = True
            self._buffer = list(self._pre_roll)
            self._pre_roll.clear()
            # В pre-roll последние _speech_count кадров — речь; более ранние
            # кадры являются контекстом до её начала.
            self._utterance_speech_frames = min(
                self._speech_count, len(self._buffer)
            )
            self._silence_count = 0
            if self._utterance_speech_frames >= self.max_frames:
                self._flush_reason = "max"
                return True
            return False

        self._buffer.append(frame)
        if is_speech:
            self._speech_count += 1
            self._utterance_speech_frames += 1
            self._silence_count = 0
        else:
            self._speech_count = 0
            self._silence_count += 1

        # конец реплики: достаточно тишины
        if self._silence_count >= self.silence_frames:
            self._flush_reason = "silence"
            return True
        # или превышен максимум длительности именно речи
        if self._utterance_speech_frames >= self.max_frames:
            self._flush_reason = "max"
            return True
        return False

    def flush(self) -> bytes:
        """Собрать WAV s16le 8 kГц mono из накопленных u8-кадров и сбросить VAD."""
        info = VadFlushInfo(
            speech_ms=self._utterance_speech_frames * FRAME_MS,
            buffered_ms=len(self._buffer) * FRAME_MS,
            trailing_silence_ms=min(
                self._silence_count, self.silence_frames
            )
            * FRAME_MS,
            reason=self._flush_reason if self._has_speech else "manual",
        )
        pcm_u8 = b"".join(self._buffer)
        self.reset()
        self._last_flush_info = info
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


__all__ = ["EnergyVAD", "VadFlushInfo"]
