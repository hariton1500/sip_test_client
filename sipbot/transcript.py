"""Хранилище текстовой ленты разговора для веб-интерфейса.

Сообщения публикуются из аудио-потока (ASR) и системных событий звонка.
UI получает их через SSE, а хранилище держит короткую историю, чтобы страница
могла восстановиться после переподключения.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

CALLER = "caller"
BOT = "bot"
SYSTEM = "system"
VALID_ROLES = (CALLER, BOT, SYSTEM)


@dataclass(frozen=True)
class TranscriptMessage:
    """Одна реплика или системное событие."""

    id: int
    call_id: str
    role: str
    text: str
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "call_id": self.call_id,
            "role": self.role,
            "text": self.text,
            "created_at": self.created_at,
        }


class TranscriptStore:
    """Потокобезопасная лента сообщений с подпиской для SSE."""

    def __init__(self, max_messages: int = 500) -> None:
        self._messages: Deque[TranscriptMessage] = deque(maxlen=max_messages)
        self._listeners: List[queue.Queue] = []
        self._lock = threading.RLock()
        self._next_id = 1

    # ------------------------------------------------------------------
    # Запись
    # ------------------------------------------------------------------
    def publish(self, role: str, text: str, call_id: str = "") -> Optional[Dict[str, Any]]:
        """Добавляет сообщение и рассылает его подписчикам.

        Пустые реплики игнорируются. Роль нормализуется: неизвестная
        превращается в ``system``.
        """
        text = " ".join(text.split())
        if not text:
            return None
        if role not in VALID_ROLES:
            role = SYSTEM

        with self._lock:
            message = TranscriptMessage(
                id=self._next_id,
                call_id=call_id,
                role=role,
                text=text,
                created_at=time.time(),
            )
            self._next_id += 1
            self._messages.append(message)
            payload = {"type": "message", **message.to_dict()}
            self._broadcast_locked(payload)
        return payload

    def clear(self) -> None:
        """Очищает историю у всех подключённых клиентов."""
        with self._lock:
            self._messages.clear()
            self._broadcast_locked({"type": "clear"})

    # ------------------------------------------------------------------
    # Чтение и подписка
    # ------------------------------------------------------------------
    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [message.to_dict() for message in self._messages]

    def subscribe(self, maxsize: int = 500) -> "queue.Queue[Dict[str, Any]]":
        listener: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._listeners.append(listener)
        return listener

    def unsubscribe(self, listener: "queue.Queue[Dict[str, Any]]") -> None:
        with self._lock:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

    @property
    def listener_count(self) -> int:
        with self._lock:
            return len(self._listeners)

    def _broadcast_locked(self, payload: Dict[str, Any]) -> None:
        for listener in list(self._listeners):
            try:
                listener.put_nowait(payload)
            except queue.Full:
                # Медленный браузер: выбрасываем самое старое и пробуем снова.
                try:
                    listener.get_nowait()
                    listener.put_nowait(payload)
                except (queue.Empty, queue.Full):
                    pass


__all__ = [
    "BOT",
    "CALLER",
    "SYSTEM",
    "TranscriptMessage",
    "TranscriptStore",
]
