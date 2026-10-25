"""Тесты записи, аудио-моста и ботов — на фейковом звонке, без сети."""

import time
import wave

import pytest

from sipbot import audio_utils as au
from sipbot.bots import EchoBot, SilenceBot, ToneBot, WavFileBot, build_bot
from sipbot.media import MediaBridge, Recorder


class _FakeState:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeCall:
    """Минимальная заглушка VoIPCall."""

    def __init__(self, frames=None, state: str = "ANSWERED") -> None:
        self.state = _FakeState(state)
        self._frames = list(frames or [])
        self.written = []
        self.read_calls = 0

    def read_audio(self, length: int = 160, blocking: bool = True) -> bytes:
        self.read_calls += 1
        if self._frames:
            return self._frames.pop(0)
        return au.SILENCE_FRAME[:length]

    def write_audio(self, data: bytes) -> None:
        self.written.append(data)


def test_recorder_writes_16bit_wav(tmp_path):
    path = tmp_path / "rec.wav"
    recorder = Recorder(path)
    for _ in range(50):
        recorder.write(au.SILENCE_FRAME)
    recorder.close()

    assert recorder.frames == 50
    assert recorder.duration_ms == 1000
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 8000
        assert wf.getnframes() == 8000
        assert wf.readframes(8000) == b"\x00\x00" * 8000


def test_recorder_close_is_idempotent(tmp_path):
    recorder = Recorder(tmp_path / "rec.wav")
    recorder.write(au.SILENCE_FRAME)
    recorder.close()
    recorder.close()
    recorder.write(au.SILENCE_FRAME)  # после close игнорируется
    assert recorder.frames == 1


def test_bridge_echoes_audio(tmp_path):
    tone = au.tone_u8(440, 20, amplitude=0.5)
    call = FakeCall(frames=[tone] * 5)
    recorder = Recorder(tmp_path / "rec.wav")
    bridge = MediaBridge(call, bot=EchoBot(), recorder=recorder)

    bridge.start()
    time.sleep(0.4)
    bridge.stop()
    recorder.close()

    assert bridge.frames_in >= 5
    assert bridge.frames_out >= 1
    assert recorder.frames >= 5
    assert tone in call.written
    assert all(len(frame) == 160 for frame in call.written)


def test_bridge_sends_silence_when_bot_is_quiet():
    call = FakeCall()
    bridge = MediaBridge(call, bot=SilenceBot())
    bridge.start()
    time.sleep(0.15)
    bridge.stop()

    assert call.written
    assert all(frame == au.SILENCE_FRAME for frame in call.written)
    assert bridge.frames_out == 0


def test_bridge_stops_when_call_ended():
    call = FakeCall(state="ENDED")
    bridge = MediaBridge(call, bot=EchoBot())
    bridge.start()
    time.sleep(0.2)

    assert not bridge.active
    assert bridge.frames_in == 0


def test_bridge_survives_bot_exception():
    class BrokenBot:
        def __init__(self) -> None:
            self.calls = 0

        def on_frame(self, frame: bytes):
            self.calls += 1
            raise RuntimeError("boom")

        def close(self) -> None:
            pass

    bot = BrokenBot()
    call = FakeCall(frames=[au.SILENCE_FRAME] * 5)
    bridge = MediaBridge(call, bot=bot)
    bridge.start()
    time.sleep(0.3)
    bridge.stop()

    assert bot.calls >= 1
    assert bridge.frames_in >= 5


def test_bridge_pads_short_bot_frames():
    class ShortBot:
        def on_frame(self, frame: bytes) -> bytes:
            return b"\x01\x02"

        def close(self) -> None:
            pass

    call = FakeCall(frames=[au.SILENCE_FRAME] * 3)
    bridge = MediaBridge(call, bot=ShortBot())
    bridge.start()
    time.sleep(0.2)
    bridge.stop()

    assert call.written
    assert all(len(frame) == 160 for frame in call.written)
    assert call.written[0].startswith(b"\x01\x02")


def test_inject_keeps_long_audio_without_drops():
    bridge = MediaBridge(FakeCall())
    first, second = bytes([1]) * 160, bytes([2]) * 160
    long_audio = first + second + b"\x03" * 160 * 298  # 6 секунд

    bridge.inject(long_audio)

    assert bridge.dropped == 0
    assert bridge._out.qsize() == 300
    assert bridge.stats()["queued"] == 300
    assert bridge._out.get_nowait() == first
    assert bridge._out.get_nowait() == second


def test_clear_outgoing_discards_pending_audio():
    bridge = MediaBridge(FakeCall())
    bridge.inject(b"\x01" * 160 * 5)

    assert bridge.clear_outgoing() == 5
    assert bridge._out.qsize() == 0
    assert bridge.clear_outgoing() == 0


def test_build_bot_variants(tmp_path):
    assert build_bot(None) is None
    assert isinstance(build_bot("silence"), SilenceBot)
    assert isinstance(build_bot("echo"), EchoBot)
    assert isinstance(build_bot("tone"), ToneBot)

    tone_bot = build_bot("tone:1500:40")
    assert isinstance(tone_bot, ToneBot)
    frames = [tone_bot.on_frame(au.SILENCE_FRAME) for _ in range(6)]
    assert all(len(frame) == 160 for frame in frames)
    assert any(frame != au.SILENCE_FRAME for frame in frames)

    wav_path = tmp_path / "greet.wav"
    au.write_wav(wav_path, au.tone_s16(1000, 100, rate=16000), 16000)
    assert isinstance(build_bot(f"wav:{wav_path}"), WavFileBot)
    assert isinstance(build_bot(f"wav:{wav_path}:loop"), WavFileBot)

    with pytest.raises(ValueError):
        build_bot("bogus")


def test_wav_bot_resamples_and_stops(tmp_path):
    path = tmp_path / "greet.wav"
    au.write_wav(path, au.tone_s16(1000, 100, rate=16000), 16000)

    bot = WavFileBot(str(path))
    frames = [bot.on_frame(au.SILENCE_FRAME) for _ in range(5)]
    assert len(frames) == 5
    assert all(len(frame) == 160 for frame in frames)
    assert any(frame != au.SILENCE_FRAME for frame in frames)
    # 100 мс = 5 кадров, дальше файл закончился
    assert bot.on_frame(au.SILENCE_FRAME) is None


def test_wav_bot_loop_repeats(tmp_path):
    path = tmp_path / "greet.wav"
    au.write_wav(path, au.tone_s16(1000, 100, rate=8000), 8000)

    bot = WavFileBot(str(path), loop=True)
    frames = [bot.on_frame(au.SILENCE_FRAME) for _ in range(11)]
    assert frames[0] == frames[5]
    assert all(frame is not None for frame in frames)
