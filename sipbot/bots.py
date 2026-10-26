"""Простые обработчики входящего аудио («боты»).

Каждый бот получает 20-мс кадры во внутреннем формате pyVoIP (u8) и может
вернуть кадр для отправки собеседнику (``None`` = ничего не отправлять).
Сюда позже встраиваются VAD/ASR/LLM/TTS — SIP-часть менять не придётся.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Protocol, runtime_checkable

from .ai_client import AiClient
from .audio_utils import (
    FRAME_BYTES,
    FRAME_MS,
    SAMPLE_RATE,
    SILENCE_BYTE,
    SILENCE_FRAME,
    iter_frames,
    read_wav,
    resample_s16,
    s16_to_u8,
    tone_u8,
)
from .billing import BillingClient
from .vad import EnergyVAD

logger = logging.getLogger(__name__)


@runtime_checkable
class Bot(Protocol):
    """Интерфейс бота."""

    def on_call_started(
        self, caller_id: str, call_id: str
    ) -> Optional[bytes]:
        """Вызывается при ответе; может вернуть байты PCM u8 для приветствия."""

    def on_call_ended(self) -> None:
        """Освобождение ресурсов, специфичных для одного звонка."""

    def on_frame(self, frame: bytes) -> Optional[bytes]:
        """Обработать входящий кадр и вернуть кадр для отправки (или None)."""

    def close(self) -> None:
        """Освободить ресурсы (вызывается один раз при уничтожении бота)."""


class _LifecycleBot:
    """Mixin: безопасные пустые реализации для on_call_started/on_call_ended."""

    def on_call_started(
        self, caller_id: str, call_id: str
    ) -> Optional[bytes]:
        return None

    def on_call_ended(self) -> None:
        pass


class SilenceBot(_LifecycleBot):
    """Ничего не отправляет (pyVoIP сам шлёт тишину)."""

    def on_frame(self, frame: bytes) -> Optional[bytes]:
        return None

    def close(self) -> None:
        pass


class EchoBot(_LifecycleBot):
    """Возвращает вход как есть — базовая проверка двустороннего аудио."""

    def on_frame(self, frame: bytes) -> Optional[bytes]:
        return frame

    def close(self) -> None:
        pass


class ToneBot(_LifecycleBot):
    """Циклически проигрывает тон: тон + пауза (проверка исходящего звука)."""

    def __init__(
        self,
        freq_hz: float = 1000.0,
        duration_ms: int = 300,
        pause_ms: int = 2700,
    ) -> None:
        tone_frames = list(
            iter_frames(tone_u8(freq_hz, duration_ms), pad_byte=SILENCE_BYTE)
        )
        pause_frames = max(0, int(pause_ms / FRAME_MS))
        self._cycle: List[bytes] = tone_frames + [SILENCE_FRAME] * pause_frames
        if not self._cycle:
            self._cycle = [SILENCE_FRAME]
        self._index = 0

    def on_frame(self, frame: bytes) -> Optional[bytes]:
        chunk = self._cycle[self._index % len(self._cycle)]
        self._index += 1
        return chunk

    def close(self) -> None:
        pass


class WavFileBot(_LifecycleBot):
    """Проигрывает WAV-файл (ресемплинг в 8 кГц), затем тишину."""

    def __init__(self, path: str, loop: bool = False) -> None:
        pcm, rate = read_wav(path)
        pcm_8k = resample_s16(pcm, rate, SAMPLE_RATE)
        self._frames: List[bytes] = list(
            iter_frames(s16_to_u8(pcm_8k), pad_byte=SILENCE_BYTE)
        ) or [SILENCE_FRAME]
        self.loop = loop
        self._index = 0
        logger.info(
            "WavFileBot: %s, %.1f с, %d кадров",
            path,
            len(self._frames) * FRAME_MS / 1000.0,
            len(self._frames),
        )

    def on_frame(self, frame: bytes) -> Optional[bytes]:
        if self._index >= len(self._frames):
            if not self.loop:
                return None
            self._index = 0
        chunk = self._frames[self._index]
        self._index += 1
        return chunk

    def close(self) -> None:
        pass


class PlaceholderBot(_LifecycleBot):
    """Заглушка входящего бота: отправляет приветствие и далее молчит.

    Пригодится для тестирования многоканального приёмника до подключения
    реального ASR/LLM/TTS.
    """

    def __init__(
        self,
        greeting_ms: int = 500,
        freq_hz: float = 800.0,
    ) -> None:
        self._greeting = tone_u8(freq_hz, greeting_ms)

    def on_call_started(
        self, caller_id: str, call_id: str
    ) -> Optional[bytes]:
        logger.info("Placeholder: входящий звонок от %s (%s)", caller_id, call_id)
        return self._greeting

    def on_frame(self, frame: bytes) -> Optional[bytes]:
        return None

    def close(self) -> None:
        pass


def build_bot(spec: Optional[str]) -> Optional[Bot]:
    """Создаёт бота по строке из CLI.

    Допустимые значения: ``silence``, ``echo``, ``tone``, ``tone:<Гц>:<мс>``,
    ``wav:<путь>``, ``wav:<путь>:loop``.
    """
    if not spec or spec == "silence":
        return SilenceBot() if spec else None
    if spec == "echo":
        return EchoBot()
    if spec == "tone":
        return ToneBot()
    if spec == "placeholder":
        return PlaceholderBot()
    if spec.startswith("placeholder:"):
        _, greeting = spec.split(":", 1)
        return PlaceholderBot(greeting_ms=int(greeting))
    if spec.startswith("tone:"):
        _, freq, duration = (spec.split(":") + ["", ""])[:3]
        return ToneBot(
            freq_hz=float(freq) if freq else 1000.0,
            duration_ms=int(duration) if duration else 300,
        )
    if spec.startswith("wav:"):
        _, rest = spec.split(":", 1)
        loop = rest.endswith(":loop")
        path = rest[: -len(":loop")] if loop else rest
        return WavFileBot(path, loop=loop)
    raise ValueError(f"Неизвестный тип бота: {spec!r}")


__all__ = [
    "Bot",
    "EchoBot",
    "PlaceholderBot",
    "SilenceBot",
    "ToneBot",
    "WavFileBot",
    "build_bot",
    "FRAME_BYTES",
]
