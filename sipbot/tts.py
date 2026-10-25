"""Синтез речи и кэш WAV для телефонного канала.

TTS-сервис принимает WAV от OpenAI-совместимого API, приводит его к
16-битному PCM, ресемплирует в 8 кГц и конвертирует во внутренний u8-формат
pyVoIP. Результат можно кэшировать на диск: приветствие не синтезируется
заново на каждый звонок.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import tempfile
import threading
import wave
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .audio_utils import SAMPLE_RATE, resample_s16, s16_to_u8

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def pcm_s16_to_wav_bytes(pcm_s16: bytes, rate: int) -> bytes:
    """Упаковать raw s16le PCM в WAV bytes (для единого дискового кэша)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm_s16)
    return buffer.getvalue()


def wav_bytes_to_pcm(data: bytes) -> Tuple[bytes, int]:
    """Декодировать WAV bytes в моно s16le PCM.

    Поддерживаются 8/16/24/32-битные PCM WAV и любое число каналов.
    TTS-провайдеры обычно отдают 16-битный моно WAV, но заголовок лучше
    проверять, а не предполагать.
    """
    if not data:
        raise ValueError("Пустой TTS-ответ")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"TTS вернул не WAV: {exc}") from exc

    if channels <= 0 or rate <= 0:
        raise ValueError("Некорректные параметры WAV от TTS")

    if width == 1:
        samples = np.frombuffer(frames, dtype=np.uint8).astype(np.int16)
        samples = (samples.astype(np.int16) - 128) * 256
    elif width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.int16)
    elif width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        value = np.where(value & 0x800000, value - 0x1000000, value)
        samples = (value >> 8).astype(np.int16)
    elif width == 4:
        samples = (np.frombuffer(frames, dtype="<i4") >> 16).astype(np.int16)
    else:
        raise ValueError(f"Неподдерживаемая разрядность TTS WAV: {width * 8} бит")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
        samples = np.clip(np.rint(samples), -32768, 32767).astype(np.int16)

    return samples.astype("<i2").tobytes(), rate


class TtsService:
    """Синтез с дисковым кэшем и приведением к формату pyVoIP."""

    def __init__(
        self,
        ai,
        cache_dir: Optional[PathLike] = None,
        max_memory_items: int = 64,
    ) -> None:
        if ai is None:
            raise ValueError("TtsService требует AiClient")
        self.ai = ai
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        if self.cache_dir is not None:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                logger.warning(
                    "Не удалось создать TTS-кэш %s, работаю без диска",
                    self.cache_dir,
                    exc_info=True,
                )
                self.cache_dir = None
        self.max_memory_items = max(1, max_memory_items)
        self._memory: "OrderedDict[str, bytes]" = OrderedDict()
        self._lock = threading.RLock()
        # Разные фразы можно синтезировать параллельно (важно для нескольких
        # линий); одинаковые — ждут один сетевой вызов.
        self._key_locks: Dict[str, threading.Lock] = {}
        self._key_locks_guard = threading.Lock()

    def audio_u8(self, text: str) -> bytes:
        """Синтезировать текст и вернуть u8 PCM 8 кГц (формат pyVoIP)."""
        text = " ".join((text or "").split())
        if not text:
            return b""

        key = self._cache_key(text)
        with self._lock:
            cached = self._memory.get(key)
            if cached is not None:
                self._memory.move_to_end(key)
                return cached

        with self._key_lock(key):
            with self._lock:
                cached = self._memory.get(key)
                if cached is not None:
                    self._memory.move_to_end(key)
                    return cached

            pcm_s16, rate = self._load_or_synthesize(key, text)
            audio = s16_to_u8(resample_s16(pcm_s16, rate, SAMPLE_RATE))
            with self._lock:
                self._memory[key] = audio
                self._memory.move_to_end(key)
                while len(self._memory) > self.max_memory_items:
                    self._memory.popitem(last=False)
            return audio

    def _key_lock(self, key: str) -> threading.Lock:
        with self._key_locks_guard:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    def preload(self, text: str) -> bytes:
        """Заранее синтезировать фразу (например, приветствие)."""
        return self.audio_u8(text)

    def _cache_key(self, text: str) -> str:
        model = str(getattr(self.ai, "tts_model", "") or "")
        voice = str(getattr(self.ai, "tts_voice", "") or "")
        payload = "\x00".join((model, voice, text)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _load_or_synthesize(self, key: str, text: str) -> Tuple[bytes, int]:
        path = self.cache_dir / f"{key}.wav" if self.cache_dir is not None else None
        if path is not None and path.is_file():
            try:
                return wav_bytes_to_pcm(path.read_bytes())
            except (OSError, ValueError):
                logger.warning("Повреждён TTS-кэш %s, синтезирую заново", path)

        raw = self.ai.synthesize(text)
        if raw[:4] == b"RIFF":
            wav_bytes = raw
            pcm_s16, rate = wav_bytes_to_pcm(wav_bytes)
        else:
            # AiTunnel для gemini-*-tts отдаёт raw pcm, а не WAV.
            rate = int(getattr(self.ai, "tts_sample_rate", 24000) or 24000)
            if len(raw) % 2:
                raw = raw[:-1]
            pcm_s16 = raw
            if not pcm_s16:
                raise ValueError("TTS вернул пустой PCM")
            wav_bytes = pcm_s16_to_wav_bytes(pcm_s16, rate)

        if path is not None:
            try:
                self._write_atomic(path, wav_bytes)
            except OSError:
                logger.warning("Не удалось сохранить TTS-кэш %s", path, exc_info=True)
        return pcm_s16, rate

    @staticmethod
    def _write_atomic(path: Path, data: bytes) -> None:
        fd = -1
        tmp_name = ""
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
            )
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        finally:
            if fd >= 0:
                os.close(fd)
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass


__all__ = ["TtsService", "pcm_s16_to_wav_bytes", "wav_bytes_to_pcm"]
