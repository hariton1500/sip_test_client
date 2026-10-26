"""Тесты конвертеров, WAV и кадров (без сети и без pyVoIP)."""

import wave

import numpy as np
import pytest

from sipbot import audio_utils as au


def test_constants():
    assert au.SAMPLE_RATE == 8000
    assert au.FRAME_MS == 20
    assert au.FRAME_SAMPLES == 160
    assert au.FRAME_BYTES == 160
    assert au.SILENCE_FRAME == b"\x80" * 160


@pytest.mark.parametrize(
    "value,expected",
    [
        (0x80, 0),  # тишина
        (0x00, -32768),  # минимум
        (0xFF, 32512),  # максимум
    ],
)
def test_u8_to_s16(value, expected):
    out = np.frombuffer(au.u8_to_s16(bytes([value])), dtype="<i2")
    assert int(out[0]) == expected


def test_u8_s16_roundtrip_is_lossless():
    data = bytes(range(256))
    assert au.s16_to_u8(au.u8_to_s16(data)) == data


def test_empty_inputs():
    for func in (
        au.u8_to_s16,
        au.s16_to_u8,
        au.ulaw_decode,
        au.ulaw_encode,
        au.alaw_decode,
        au.alaw_encode,
    ):
        assert func(b"") == b""


def test_ulaw_silence():
    # pyVoIP: bias(-128) -> lin2ulaw; тишина в u8 = 0x80 -> 0xFF
    assert au.ulaw_encode(b"\x80") == b"\xff"
    assert au.ulaw_decode(b"\xff") == b"\x80"


def test_g711_lengths_preserved():
    tone = au.tone_u8(440, 20, amplitude=0.5)
    assert len(au.ulaw_encode(tone)) == len(tone)
    assert len(au.ulaw_decode(au.ulaw_encode(tone))) == len(tone)
    assert len(au.alaw_encode(tone)) == len(tone)
    assert len(au.alaw_decode(au.alaw_encode(tone))) == len(tone)


def test_ulaw_roundtrip_error_is_small():
    tone = au.tone_u8(440, 40, amplitude=0.5)
    decoded = au.ulaw_decode(au.ulaw_encode(tone))
    original = np.frombuffer(au.u8_to_s16(tone), dtype="<i2").astype(int)
    restored = np.frombuffer(au.u8_to_s16(decoded), dtype="<i2").astype(int)
    assert np.abs(original - restored).max() < 5000


def test_resample_s16_rates():
    pcm = au.tone_s16(440, 100, rate=8000)
    assert len(au.resample_s16(pcm, 8000, 16000)) == 2 * len(pcm)
    assert len(au.resample_s16(pcm, 8000, 4000)) == len(pcm) // 2
    assert au.resample_s16(pcm, 8000, 8000) == pcm
    assert au.resample_s16(b"", 8000, 16000) == b""


def test_resample_suppresses_aliasing():
    """5 кГц при переходе 24k->8k должны быть задавлены ФНЧ, а не завернуться в 3 кГц."""
    pcm = au.tone_s16(5000, 100, rate=24000, amplitude=0.8)
    out = np.frombuffer(au.resample_s16(pcm, 24000, 8000), dtype="<i2")
    center = out[80:-80]  # без краевых эффектов фильтра
    assert np.abs(center).max() < 0.1 * 32767


def test_resample_keeps_speech_band():
    pcm = au.tone_s16(1000, 100, rate=24000, amplitude=0.5)
    out = np.frombuffer(au.resample_s16(pcm, 24000, 8000), dtype="<i2")
    center = out[80:-80]
    assert np.abs(center).max() > 0.45 * 32767


def test_resample_rejects_bad_rate():
    with pytest.raises(ValueError):
        au.resample_s16(au.tone_s16(440, 20), 0, 8000)


def test_resample_u8_keeps_silence():
    assert au.resample_u8(au.SILENCE_FRAME, 8000, 16000) == b"\x80" * 320


def test_iter_frames_pads_last_frame():
    frames = list(au.iter_frames(bytes(range(200)), 160))
    assert len(frames) == 2
    assert all(len(f) == 160 for f in frames)
    assert frames[1][-40:] == b"\x00" * 40

    padded = list(au.iter_frames(b"\x01" * 170, 160, pad_byte=au.SILENCE_BYTE))
    assert padded[1] == b"\x01" * 10 + b"\x80" * 150


def test_iter_frames_edge_cases():
    assert list(au.iter_frames(b"")) == []
    with pytest.raises(ValueError):
        list(au.iter_frames(b"123", 0))


def test_wav_roundtrip(tmp_path):
    pcm = au.tone_s16(1000, 100)
    path = tmp_path / "tone.wav"
    au.write_wav(path, pcm)

    restored, rate = au.read_wav(path)
    assert rate == au.SAMPLE_RATE
    assert restored == pcm

    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == au.SAMPLE_RATE
        assert wf.getnframes() == 800


def test_read_wav_downmixes_stereo(tmp_path):
    left = np.full(800, 1000, dtype="<i2")
    right = np.full(800, -1000, dtype="<i2")
    stereo = np.stack([left, right], axis=1).tobytes()
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(stereo)

    mono, rate = au.read_wav(path)
    assert rate == 16000
    assert len(mono) == 1600
    assert np.abs(np.frombuffer(mono, dtype="<i2")).max() <= 1


def test_read_wav_rejects_non_16bit(tmp_path):
    path = tmp_path / "u8.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(8000)
        wf.writeframes(b"\x80" * 800)

    with pytest.raises(ValueError, match="16-битный"):
        au.read_wav(path)


def test_tone_and_rms():
    tone = au.tone_u8(1000, 20, amplitude=0.5)
    assert len(tone) == au.FRAME_BYTES
    assert 0.2 < au.rms_u8(tone) < 0.6
    assert au.rms_u8(au.SILENCE_FRAME) == 0.0
    assert au.rms_u8(b"") == 0.0
    assert au.duration_ms_u8(au.SILENCE_FRAME) == 20.0


def test_tone_zero_duration():
    assert au.tone_s16(1000, 0) == b""
    assert au.tone_u8(1000, 0) == b""
    assert au.silence_s16(20) == b"\x00\x00" * 160
