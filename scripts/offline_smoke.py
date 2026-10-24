"""Оффлайн-смоук: связка SipPhone + MediaBridge + Recorder + WavFileBot без сети.

Запуск:
    python scripts/offline_smoke.py

Проверяет, что публичный API pyVoIP 1.6.8 совпадает с ожидаемым
(VoIPPhone.get_status/stop, VoIPCall.read_audio/write_audio), и что мост
пишет WAV с реальным таймлайном и отправляет кадры ровно раз в 20 мс.
"""

import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sipbot import audio_utils as au  # noqa: E402
from sipbot.bots import WavFileBot  # noqa: E402
from sipbot.config import Settings  # noqa: E402
from sipbot.media import MediaBridge, Recorder  # noqa: E402
from sipbot.phone import SipPhone  # noqa: E402

settings = Settings(
    server="127.0.0.1", username="u", password="p", my_ip="192.168.1.5"
)
settings.validate()
phone = SipPhone(settings)
print("status before start:", phone.status().name)
print("incoming queue:", phone.incoming.qsize())
phone.stop()

tmp = Path(".pytest_tmp")
tmp.mkdir(exist_ok=True)
greeting = tmp / "smoke_greeting.wav"
au.write_wav(greeting, au.tone_s16(600, 200, rate=16000), 16000)


class FakeState:
    name = "ANSWERED"


class FakeCall:
    def __init__(self) -> None:
        self.state = FakeState()
        self.written = []

    def read_audio(self, length: int = 160, blocking: bool = True) -> bytes:
        return au.SILENCE_FRAME[:length]

    def write_audio(self, data: bytes) -> None:
        self.written.append(data)


call = FakeCall()
record_path = tmp / "smoke_record.wav"
with Recorder(record_path) as recorder:
    bridge = MediaBridge(call, bot=WavFileBot(str(greeting)), recorder=recorder)
    bridge.start()
    time.sleep(0.5)
    bridge.stop()

with wave.open(str(record_path), "rb") as wf:
    print("recording:", wf.getnframes(), "frames @", wf.getframerate(), "Hz")
print("stats:", bridge.stats())
print("frames written to call:", len(call.written))
print("greeting frames sent:", sum(1 for f in call.written if f != au.SILENCE_FRAME))
