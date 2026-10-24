"""Работа со звуком для pyVoIP.

pyVoIP внутри использует **8-битный unsigned PCM** (8000 Гц, моно):

* 1 сэмпл = 1 байт;
* 160 байт = 20 мс;
* 0x80 (128) = тишина.

WAV-файлы, ASR и TTS обычно работают с 16-битным signed PCM, поэтому здесь
собраны конвертеры, ресемплинг, чтение/запись WAV и нарезка на 20-мс кадры.
"""

from __future__ import annotations

import warnings
import wave
from pathlib import Path
from typing import Iterator, Tuple, Union

import numpy as np

SAMPLE_RATE = 8000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 160 сэмплов
FRAME_BYTES = FRAME_SAMPLES  # во внутреннем формате pyVoIP 1 байт на сэмпл
SILENCE_BYTE = 0x80
SILENCE_FRAME = bytes([SILENCE_BYTE]) * FRAME_BYTES

PathLike = Union[str, Path]

with warnings.catch_warnings():
    # audioop помечен deprecated в 3.11 и удалён в 3.13 (там нужен audioop-lts).
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        import audioop  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - зависит от версии Python
        raise ModuleNotFoundError(
            "Модуль 'audioop' недоступен (Python >= 3.13). "
            "Установите пакет 'audioop-lts'."
        ) from exc


# --------------------------------------------------------------------------
# Конвертация между внутренним форматом pyVoIP (u8) и обычным PCM (s16le)
# --------------------------------------------------------------------------
def u8_to_s16(pcm: bytes) -> bytes:
    """u8 (0x80 = тишина) -> s16le."""
    if not pcm:
        return b""
    data = np.frombuffer(pcm, dtype=np.uint8).astype(np.int16) - SILENCE_BYTE
    return (data * 256).astype("<i2").tobytes()


def s16_to_u8(pcm: bytes) -> bytes:
    """s16le -> u8 (0x80 = тишина)."""
    if not pcm:
        return b""
    data = np.frombuffer(pcm, dtype="<i2").astype(np.int32) >> 8
    return np.clip(data + SILENCE_BYTE, 0, 255).astype(np.uint8).tobytes()


# --------------------------------------------------------------------------
# G.711 (повторяет внутреннюю логику pyVoIP: audioop + bias, width=1)
# --------------------------------------------------------------------------
def ulaw_decode(payload: bytes) -> bytes:
    """PCMU payload (RTP) -> u8 PCM."""
    return audioop.bias(audioop.ulaw2lin(payload, 1), 1, 128)


def ulaw_encode(pcm_u8: bytes) -> bytes:
    """u8 PCM -> PCMU payload (RTP)."""
    return audioop.lin2ulaw(audioop.bias(pcm_u8, 1, -128), 1)


def alaw_decode(payload: bytes) -> bytes:
    """PCMA payload (RTP) -> u8 PCM."""
    return audioop.bias(audioop.alaw2lin(payload, 1), 1, 128)


def alaw_encode(pcm_u8: bytes) -> bytes:
    """u8 PCM -> PCMA payload (RTP)."""
    return audioop.lin2alaw(audioop.bias(pcm_u8, 1, -128), 1)


# --------------------------------------------------------------------------
# Ресемплинг и генерация
# --------------------------------------------------------------------------
def resample_s16(pcm: bytes, src_rate: int, dst_rate: int = SAMPLE_RATE) -> bytes:
    """Линейный ресемплинг s16le (без ФНЧ — для тест-клиента достаточно)."""
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("Частоты дискретизации должны быть положительными")
    if src_rate == dst_rate or not pcm:
        return pcm

    data = np.frombuffer(pcm, dtype="<i2")
    n_out = max(1, int(round(data.size * dst_rate / src_rate)))
    x_old = np.linspace(0.0, 1.0, data.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    out = np.interp(x_new, x_old, data.astype(np.float64))
    return np.clip(np.rint(out), -32768, 32767).astype("<i2").tobytes()


def resample_u8(pcm: bytes, src_rate: int, dst_rate: int = SAMPLE_RATE) -> bytes:
    """Линейный ресемплинг во внутреннем формате pyVoIP (u8)."""
    return s16_to_u8(resample_s16(u8_to_s16(pcm), src_rate, dst_rate))


def tone_s16(
    freq_hz: float = 1000.0,
    duration_ms: int = 300,
    rate: int = SAMPLE_RATE,
    amplitude: float = 0.3,
) -> bytes:
    """Синусоидальный тон в s16le."""
    if duration_ms <= 0:
        return b""
    n = int(rate * duration_ms / 1000)
    t = np.arange(n, dtype=np.float64) / rate
    samples = np.sin(2.0 * np.pi * freq_hz * t) * (32767.0 * amplitude)
    return np.rint(samples).astype("<i2").tobytes()


def tone_u8(
    freq_hz: float = 1000.0,
    duration_ms: int = 300,
    rate: int = SAMPLE_RATE,
    amplitude: float = 0.3,
) -> bytes:
    """Синусоидальный тон во внутреннем формате pyVoIP (u8)."""
    return s16_to_u8(tone_s16(freq_hz, duration_ms, rate, amplitude))


def silence_s16(duration_ms: int, rate: int = SAMPLE_RATE) -> bytes:
    """Тишина в s16le."""
    return b"\x00\x00" * int(rate * duration_ms / 1000)


# --------------------------------------------------------------------------
# WAV
# --------------------------------------------------------------------------
def read_wav(path: PathLike) -> Tuple[bytes, int]:
    """Читает WAV (16 бит), возвращает (s16le моно, sample_rate)."""
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(
                f"{path}: поддерживается только 16-битный PCM WAV "
                f"(sampwidth={wf.getsampwidth()})"
            )
        channels = wf.getnchannels()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if channels == 1:
        return frames, rate

    data = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
    mono = data.mean(axis=1)
    return np.clip(np.rint(mono), -32768, 32767).astype("<i2").tobytes(), rate


def write_wav(path: PathLike, pcm_s16: bytes, rate: int = SAMPLE_RATE) -> None:
    """Пишет моно WAV 16 бит."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm_s16)


# --------------------------------------------------------------------------
# Кадры и измерения
# --------------------------------------------------------------------------
def iter_frames(
    pcm: bytes, frame_bytes: int = FRAME_BYTES, pad_byte: int = 0
) -> Iterator[bytes]:
    """Нарезает поток на кадры фиксированного размера.

    Последний неполный кадр дополняется до ``frame_bytes`` байтом ``pad_byte``
    (по умолчанию 0 — для s16; для формата pyVoIP используйте SILENCE_BYTE).
    """
    if frame_bytes <= 0:
        raise ValueError("frame_bytes должен быть положительным")
    for offset in range(0, len(pcm), frame_bytes):
        chunk = pcm[offset : offset + frame_bytes]
        if len(chunk) < frame_bytes:
            chunk = chunk + bytes([pad_byte]) * (frame_bytes - len(chunk))
        yield chunk


def rms_u8(pcm_u8: bytes) -> float:
    """Уровень сигнала 0.0..1.0 для кадра во внутреннем формате pyVoIP."""
    if not pcm_u8:
        return 0.0
    return audioop.rms(u8_to_s16(pcm_u8), 2) / 32768.0


def duration_ms_u8(pcm_u8: bytes, rate: int = SAMPLE_RATE) -> float:
    """Длительность фрагмента во внутреннем формате pyVoIP, мс."""
    return len(pcm_u8) * 1000.0 / rate
