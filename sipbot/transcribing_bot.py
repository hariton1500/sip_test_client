"""Бот-обёртка, которая распознаёт речь абонента и публикует текст в UI.

Аудио-поток остаётся в реальном времени: VAD работает в читающем потоке
``MediaBridge``, а сетевой вызов ASR выполняется в отдельном демоне-потоке.
Поэтому распознавание не задерживает отправку/приём RTP.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

from .ai_client import AiClient, AiClientError
from .bots import Bot
from .transcript import CALLER, SYSTEM, TranscriptStore
from .vad import EnergyVAD

logger = logging.getLogger(__name__)


class TranscribingBot:
    """Делегирует аудио боту и параллельно распознаёт реплики абонента."""

    def __init__(
        self,
        delegate: Optional[Bot],
        store: TranscriptStore,
        ai: Optional[AiClient],
        vad: Optional[EnergyVAD] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        max_pending: int = 20,
    ) -> None:
        self._delegate = delegate
        self._store = store
        self._ai = ai
        self._vad = vad or EnergyVAD()
        self._language = language
        self._prompt = prompt
        self._call_id = ""
        self._closed = threading.Event()
        self._queue: "queue.Queue[tuple[str, bytes]]" = queue.Queue(
            maxsize=max_pending
        )
        self._worker: Optional[threading.Thread] = None
        if ai is not None:
            self._worker = threading.Thread(
                target=self._transcribe_loop,
                name="sipbot-asr",
                daemon=True,
            )
            self._worker.start()

    # ------------------------------------------------------------------
    # Bot API
    # ------------------------------------------------------------------
    def on_call_started(
        self, caller_id: str, call_id: str
    ) -> Optional[bytes]:
        self._call_id = call_id
        self._vad.reset()
        self._store.publish(
            SYSTEM,
            f"Входящий звонок: {caller_id or 'неизвестный'}",
            call_id=call_id,
        )
        if self._delegate is None:
            return None
        return self._delegate.on_call_started(caller_id, call_id)

    def on_frame(self, frame: bytes) -> Optional[bytes]:
        if frame and self._ai is not None and not self._closed.is_set():
            try:
                if self._vad.feed(frame):
                    self._queue_utterance(self._vad.flush())
            except Exception:  # noqa: BLE001 - VAD не должен рвать звонок
                logger.exception("Ошибка VAD")
                self._vad.reset()
        if self._delegate is None:
            return None
        return self._delegate.on_frame(frame)

    def on_call_ended(self) -> None:
        if self._ai is not None and self._vad.has_speech:
            self._queue_utterance(self._vad.flush())
        else:
            self._vad.reset()
        self._store.publish(
            SYSTEM,
            "Звонок завершён",
            call_id=self._call_id,
        )
        if self._delegate is not None:
            self._delegate.on_call_ended()

    def close(self, grace: float = 1.0) -> None:
        self._closed.set()
        self._vad.close()
        if self._delegate is not None:
            self._delegate.close()
        # Даём ASR короткий шанс дораспознать реплику, поставленную
        # on_call_ended(). Поток демон, поэтому зависание приложения
        # по-прежнему невозможно.
        worker = self._worker
        if worker is not None and worker.is_alive() and grace > 0:
            worker.join(grace)

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------
    def _queue_utterance(self, wav_bytes: bytes) -> None:
        if not wav_bytes or self._ai is None:
            return
        try:
            self._queue.put_nowait((self._call_id, wav_bytes))
        except queue.Full:
            logger.warning("Очередь ASR переполнена, реплика пропущена")
            self._store.publish(
                SYSTEM,
                "Очередь распознавания переполнена, реплика пропущена",
                call_id=self._call_id,
            )

    def _transcribe_loop(self) -> None:
        while True:
            try:
                call_id, wav_bytes = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._closed.is_set():
                    return
                continue
            # Важно: уже полученную реплику обрабатываем даже после close().
            # Иначе последняя фраза при быстром отбое теряется.
            try:
                assert self._ai is not None
                text = self._ai.transcribe(
                    wav_bytes,
                    language=self._language,
                    prompt=self._prompt,
                )
            except AiClientError as exc:
                logger.warning("ASR не удался: %s", exc)
                self._store.publish(
                    SYSTEM,
                    f"Не удалось распознать реплику: {exc}",
                    call_id=call_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Неожиданная ошибка ASR")
                self._store.publish(
                    SYSTEM,
                    f"Ошибка распознавания: {exc}",
                    call_id=call_id,
                )
            else:
                text = (text or "").strip()
                if text:
                    self._store.publish(CALLER, text, call_id=call_id)
                    logger.info("ASR [%s]: %s", call_id, text)
            finally:
                self._queue.task_done()


__all__ = ["TranscribingBot"]
