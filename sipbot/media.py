"""Мост между звонком pyVoIP и ботом.

Схема:

    reader thread:  call.read_audio(160, blocking=False) -> Recorder + Bot.on_frame
    writer thread:  outgoing queue -> call.write_audio(...) сразу, без своего темпа

Чтение неблокирующее и пейсится ровно 20 мс на кадр, поэтому:

* запись разговора сохраняет реальный таймлайн (тишина тоже пишется);
* бот получает равномерный поток кадров, что важно для VAD/ASR.

Исходящий поток не пейсится дважды: темп задаёт pyVoIP (``RTPClient.trans``),
а writer лишь не даёт очереди убежать вперёд больше чем на 100 мс. Раньше
writer тоже спал по 20 мс, и из-за дрожания таймера Windows пакеты то
опаздывали (pyVoIP вставлял тишину — речь «рвалась»), то уходили с опережением,
из-за чего росла задержка и barge-in переставал останавливать звук сразу.
"""

from __future__ import annotations

import logging
import queue
import threading
import wave
from pathlib import Path
from typing import Dict, Optional, Union

from .audio_utils import (
    FRAME_BYTES,
    FRAME_MS,
    SAMPLE_RATE,
    SILENCE_BYTE,
    rms_u8,
    u8_to_s16,
)
from .bots import Bot
from .timing import now, wait_until

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

#: Максимальный запас аудио, отданного в pyVoIP вперёд реального времени.
#: Ограничивает задержку barge-in и рост очереди RTPPacketManager.
MAX_LEAD_SECONDS = 0.1


class Recorder:
    """Потоковая запись входящего звука в WAV (моно, 16 бит, 8 кГц)."""

    def __init__(self, path: PathLike, rate: int = SAMPLE_RATE) -> None:
        self.path = Path(path)
        self.rate = rate
        self.frames = 0
        self._closed = False
        self._wave = wave.open(str(self.path), "wb")
        self._wave.setnchannels(1)
        self._wave.setsampwidth(2)
        self._wave.setframerate(rate)

    def write(self, frame_u8: bytes) -> None:
        if self._closed:
            return
        self._wave.writeframes(u8_to_s16(frame_u8))
        self.frames += 1

    @property
    def duration_ms(self) -> float:
        return self.frames * FRAME_MS

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._wave.close()
        logger.info(
            "Запись сохранена: %s (%.1f с)", self.path, self.duration_ms / 1000.0
        )

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class MediaBridge:
    """Гоняет аудио между звонком и ботом в двух фоновых потоках."""

    def __init__(
        self,
        call,
        bot: Optional[Bot] = None,
        recorder: Optional[Recorder] = None,
        frame_bytes: int = FRAME_BYTES,
    ) -> None:
        self._call = call
        self._bot = bot
        self._recorder = recorder
        self._frame_bytes = frame_bytes
        self._pace = FRAME_MS / 1000.0
        # Очередь не ограничена: приветствие/TTS может быть длиннее секунды.
        # Раньше maxsize=50 обрезал всё, что не успевало уйти за 1 секунду.
        self._out: "queue.Queue[bytes]" = queue.Queue()
        self._stop = threading.Event()
        self._threads = []

        self.frames_in = 0
        self.frames_out = 0
        self.dropped = 0
        self.last_level = 0.0

    # ------------------------------------------------------------------
    # Управление
    # ------------------------------------------------------------------
    def inject(self, data: bytes) -> None:
        """Поставить все кадры аудио в очередь без потерь.

        Метод можно вызывать до ``start()``: очередь не ограничена, а writer
        после запуска отдаст кадры сразу, но не больше чем на 100 мс вперёд
        реального времени (остальное придержит).
        """
        for offset in range(0, len(data), self._frame_bytes):
            frame = data[offset : offset + self._frame_bytes]
            self._put_out(frame)

    def clear_outgoing(self) -> int:
        """Убрать ещё не отправленное аудио (например, при перебивании).

        Возвращает количество отброшенных кадров.
        """
        cleared = 0
        while True:
            try:
                self._out.get_nowait()
            except queue.Empty:
                break
            cleared += 1
        return cleared

    def start(self) -> None:
        for target, name in (
            (self._read_loop, "sipbot-reader"),
            (self._write_loop, "sipbot-writer"),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout)
        self._threads.clear()

    @property
    def active(self) -> bool:
        return any(thread.is_alive() for thread in self._threads)

    def stats(self) -> Dict[str, float]:
        return {
            "frames_in": self.frames_in,
            "frames_out": self.frames_out,
            "dropped": self.dropped,
            "queued": self._out.qsize(),
            "level": round(self.last_level, 4),
        }

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------
    def _call_alive(self) -> bool:
        state = getattr(self._call, "state", None)
        name = getattr(state, "name", None) or str(state)
        return name in ("ANSWERED", "DIALING", "RINGING")

    @staticmethod
    def _fit(frame: bytes, size: int) -> bytes:
        if len(frame) == size:
            return frame
        if len(frame) > size:
            return frame[:size]
        return frame + bytes([SILENCE_BYTE]) * (size - len(frame))

    def _put_out(self, frame: bytes) -> None:
        # Очередь не ограничена по размеру, поэтому кадры не теряются.
        self._out.put(self._fit(frame, self._frame_bytes))

    def _read_loop(self) -> None:
        next_tick = now()
        while not self._stop.is_set():
            if not self._call_alive():
                logger.info("Звонок завершён, чтение аудио остановлено")
                break
            try:
                frame = self._call.read_audio(self._frame_bytes, blocking=False)
            except Exception:  # noqa: BLE001 - стек pyVoIP кидает разные ошибки
                logger.exception("Ошибка чтения аудио из звонка")
                break

            if frame:
                frame = self._fit(frame, self._frame_bytes)
                self.frames_in += 1
                self.last_level = rms_u8(frame)
                if self._recorder is not None:
                    self._recorder.write(frame)
                if self._bot is not None:
                    try:
                        reply = self._bot.on_frame(frame)
                    except Exception:  # noqa: BLE001
                        logger.exception("Бот упал на обработке кадра")
                        reply = None
                    if reply:
                        self._put_out(reply)

            next_tick += self._pace
            current = now()
            if next_tick < current - 0.2:
                # отстали (например, при старте) — синхронизируемся заново
                next_tick = current
            wait_until(next_tick)

    def _write_loop(self) -> None:
        """Отдаёт кадры в pyVoIP сразу, ограничивая запас сотней миллисекунд.

        Темп передачи задаёт ``RTPClient.trans``: второй пейсер здесь только
        добавлял дрожание (то недосыл — pyVoIP вставлял тишину, то пересыл —
        росла задержка). Когда бот молчит, кадры не пишутся вовсе: pyVoIP сам
        отправляет тишину, пока очередь пуста.
        """
        audio_clock = 0.0
        while not self._stop.is_set():
            try:
                frame: Optional[bytes] = self._out.get(timeout=0.1)
            except queue.Empty:
                audio_clock = 0.0
                if not self._call_alive():
                    break
                continue

            if not self._call_alive():
                break
            current = now()
            if audio_clock < current:
                audio_clock = current
            if audio_clock - current > MAX_LEAD_SECONDS:
                wait_until(audio_clock - MAX_LEAD_SECONDS)

            try:
                self._call.write_audio(frame)
                self.frames_out += 1
            except Exception:  # noqa: BLE001
                logger.exception("Ошибка отправки аудио в звонок")
                break
            audio_clock += self._pace


__all__ = ["MediaBridge", "Recorder"]
