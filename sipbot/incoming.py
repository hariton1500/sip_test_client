"""Многоканальный приёмник входящих SIP-вызовов.

Схема:

    SipPhone.callCallback -> IncomingDispatcher.handle_incoming(call)
        -> CallSession.answer() -> MediaBridge + Bot + Recorder
        -> фоновый watcher ждёт CallState.ENDED -> cleanup()

Каждый вызов живёт в отдельном CallSession; количество одновременных
сессий ограничено ``Settings.max_incoming_lines``. При превышении лимита
новый вызов сразу получает ``486 Busy Here``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from pyVoIP.VoIP import CallState, VoIPCall

from .bots import Bot
from .config import Settings
from .media import MediaBridge, Recorder
from .phone import SipPhone

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


@dataclass(frozen=True)
class CallerInfo:
    """Информация о входящем звонке."""

    number: str
    display: str
    call_id: str


def caller_info(call: VoIPCall) -> CallerInfo:
    """Достаёт номер, display-name и Call-ID из входящего INVITE."""
    request = getattr(call, "request", None)
    headers = getattr(request, "headers", {}) or {}
    from_header = headers.get("From") or {}
    number = from_header.get("number") or "unknown"
    display = from_header.get("caller") or ""
    call_id = getattr(call, "call_id", "?") or "?"
    return CallerInfo(number=number, display=display, call_id=call_id)


class CallSession:
    """Одна входящая линия: ответ, медиа-мост, запись, бот, cleanup."""

    def __init__(
        self,
        call: VoIPCall,
        phone: SipPhone,
        bot: Optional[Bot] = None,
        recorder: Optional[Recorder] = None,
        on_finished: Optional[Callable[["CallSession"], None]] = None,
    ) -> None:
        self.call = call
        self.phone = phone
        self.bot = bot
        self.recorder = recorder
        self.on_finished = on_finished
        self.info = caller_info(call)
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self._bridge: Optional[MediaBridge] = None
        self._watcher: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._finished = False

    def answer(self) -> bool:
        """Отвечает на вызов. Возвращает True, если удалось."""
        if not self.phone.answer(self.call):
            return False
        self.started_at = time.monotonic()
        logger.info("Вызов принят: %s", self.info)
        return True

    def start(self) -> None:
        """Запускает медиа-мост и watcher завершения вызова."""
        with self._lock:
            if self._bridge is not None:
                return

            greeting: Optional[bytes] = None
            if self.bot is not None:
                try:
                    greeting = self.bot.on_call_started(
                        caller_id=self.info.number,
                        call_id=self.info.call_id,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Ошибка старта бота")

            self._bridge = MediaBridge(
                self.call,
                bot=self.bot,
                recorder=self.recorder,
            )
            if greeting:
                self._bridge.inject(greeting)
            self._bridge.start()

            self._watcher = threading.Thread(
                target=self._watch_loop,
                name=f"sipbot-watcher-{self.info.call_id}",
                daemon=True,
            )
            self._watcher.start()

    def stop(self) -> None:
        """Синхронно завершает сессию (используется при shutdown)."""
        self._finish(reason="manual")

    def _watch_loop(self) -> None:
        """Ждёт, пока абонент положит трубку, и чистит ресурсы."""
        try:
            while True:
                state = getattr(self.call, "state", None)
                if state is CallState.ENDED:
                    break
                time.sleep(0.1)
        except Exception:  # noqa: BLE001
            logger.exception("Ошибка в watcher звонка %s", self.info.call_id)
        finally:
            self._finish(reason="ended")

    def _finish(self, reason: str) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True

        self.ended_at = time.monotonic()
        duration = 0.0
        if self.started_at is not None:
            duration = self.ended_at - self.started_at

        logger.info(
            "Звонок завершён: %s (reason=%s, duration=%.1f с)",
            self.info,
            reason,
            duration,
        )

        if self._bridge is not None:
            self._bridge.stop()

        if self.recorder is not None:
            self.recorder.close()

        if self.bot is not None:
            try:
                self.bot.on_call_ended()
                self.bot.close()
            except Exception:  # noqa: BLE001
                logger.exception("Ошибка остановки бота")

        # Если звонок всё ещё не сброшен — принудительно положим трубку.
        try:
            if getattr(self.call, "state", None) is CallState.ANSWERED:
                self.phone.hangup(self.call)
        except Exception:  # noqa: BLE001
            logger.exception("Ошибка при принудительном сбросе звонка")
        finally:
            if self.on_finished is not None:
                try:
                    self.on_finished(self)
                except Exception:  # noqa: BLE001
                    logger.exception("Ошибка в on_finished")


class IncomingDispatcher:
    """Распределяет входящие вызовы по каналам и следит за лимитом."""

    def __init__(
        self,
        phone: SipPhone,
        settings: Settings,
        bot_factory: Callable[[], Optional[Bot]],
        record_dir: Optional[PathLike] = None,
    ) -> None:
        self.phone = phone
        self.settings = settings
        self.bot_factory = bot_factory
        self.record_dir = Path(record_dir) if record_dir else None
        self._sessions: Dict[str, CallSession] = {}
        self._lock = threading.RLock()

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def handle_incoming(self, call: VoIPCall) -> None:
        """Точка входа из SipPhone.callCallback."""
        info = caller_info(call)

        with self._lock:
            if self.active_count >= self.settings.max_incoming_lines:
                logger.warning(
                    "Достигнут лимит линий (%d), отклоняю %s",
                    self.settings.max_incoming_lines,
                    info,
                )
                self.phone.deny(call)
                return

            if info.call_id in self._sessions:
                logger.warning("Повторный INVITE для %s, игнорирую", info)
                return

            session = self._create_session(call)
            self._sessions[info.call_id] = session

        if not session.answer():
            self._remove_session(info.call_id)
            return

        session.start()

    def shutdown(self, timeout: float = 5.0) -> None:
        """Останавливает все активные сессии и ждёт завершения."""
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.stop()
        deadline = time.monotonic() + timeout
        for session in sessions:
            remaining = deadline - time.monotonic()
            watcher = getattr(session, "_watcher", None)
            if watcher is not None and watcher.is_alive() and remaining > 0:
                watcher.join(remaining)

    def _create_session(self, call: VoIPCall) -> CallSession:
        info = caller_info(call)
        recorder: Optional[Recorder] = None
        if self.record_dir is not None:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{info.call_id.replace(':', '_')}.wav"
            recorder = Recorder(self.record_dir / filename)

        bot = self.bot_factory()
        return CallSession(
            call=call,
            phone=self.phone,
            bot=bot,
            recorder=recorder,
            on_finished=self._on_session_finished,
        )

    def _on_session_finished(self, session: CallSession) -> None:
        self._remove_session(session.info.call_id)

    def _remove_session(self, call_id: str) -> None:
        with self._lock:
            self._sessions.pop(call_id, None)


__all__ = [
    "CallerInfo",
    "CallSession",
    "IncomingDispatcher",
    "caller_info",
]

