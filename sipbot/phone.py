"""Обёртка над pyVoIP: регистрация, звонки, ожидание состояний."""

from __future__ import annotations

import logging
import queue
import time
from typing import Callable, Optional

from pyVoIP.VoIP import (
    CallState,
    InvalidStateError,
    PhoneStatus,
    VoIPCall,
    VoIPPhone,
)

from . import pyvoip_fixes
from .config import Settings

logger = logging.getLogger(__name__)


def normalize_target(target: str, server: Optional[str] = None) -> str:
    """Приводит цель звонка к номеру/логину.

    pyVoIP сам формирует ``INVITE sip:<number>@<server>``, поэтому полный URI
    передавать нельзя — иначе получится ``sip:1001@host@server``.
    """
    value = target.strip()
    if not value:
        raise ValueError("Пустой номер/URI для звонка")

    lowered = value.lower()
    for prefix in ("sips:", "sip:", "tel:"):
        if lowered.startswith(prefix):
            value = value[len(prefix) :]
            break

    if "@" in value:
        number, _, host = value.partition("@")
        if server and host and host.split(":")[0] != server:
            logger.warning(
                "Хост %r из цели игнорируется: используется SIP_SERVER=%r",
                host,
                server,
            )
        value = number

    if not value:
        raise ValueError(f"Не удалось извлечь номер из {target!r}")
    return value


class SipPhone:
    """SIP-телефон с очередью входящих звонков."""

    def __init__(
        self,
        settings: Settings,
        on_incoming: Optional[Callable[[VoIPCall], None]] = None,
    ) -> None:
        self.settings = settings
        self._on_incoming = on_incoming
        self.incoming: "queue.Queue[VoIPCall]" = queue.Queue()
        # OPTIONS-keepalive АТС и устойчивая регистрация: см. pyvoip_fixes
        pyvoip_fixes.install()
        self._phone = VoIPPhone(
            server=settings.server,
            port=settings.port,
            username=settings.username,
            password=settings.password,
            myIP=settings.my_ip,
            callCallback=self._handle_incoming,
            sipPort=settings.sip_port,
            rtpPortLow=settings.rtp_port_low,
            rtpPortHigh=settings.rtp_port_high,
        )

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------
    def start(
        self,
        on_incoming: Optional[Callable[[VoIPCall], None]] = None,
    ) -> None:
        if on_incoming is not None:
            self._on_incoming = on_incoming
        logger.info("Запуск SIP-клиента: %s", self.settings.describe())
        self._phone.start()

    def stop(self) -> None:
        try:
            self._phone.stop()
        except Exception:  # noqa: BLE001 - pyVoIP может кинуть при закрытом сокете
            logger.exception("Ошибка при остановке SIP-клиента")

    def status(self) -> PhoneStatus:
        return self._phone.get_status()

    def wait_registered(
        self, timeout: float = 20.0, interval: float = 0.2
    ) -> PhoneStatus:
        """Ждёт REGISTERED или FAILED; по таймауту возвращает текущий статус."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = self.status()
            if current in (PhoneStatus.REGISTERED, PhoneStatus.FAILED):
                return current
            time.sleep(interval)
        return self.status()

    # ------------------------------------------------------------------
    # Входящие
    # ------------------------------------------------------------------
    def _handle_incoming(self, call: VoIPCall) -> None:
        logger.info("Входящий звонок: %s", describe_caller(call))
        self.incoming.put(call)
        if self._on_incoming is not None:
            try:
                self._on_incoming(call)
            except Exception:  # noqa: BLE001
                logger.exception("Ошибка в обработчике входящего звонка")

    def next_call(self, timeout: Optional[float] = None) -> Optional[VoIPCall]:
        try:
            return self.incoming.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Звонки
    # ------------------------------------------------------------------
    def dial(self, target: str) -> VoIPCall:
        number = normalize_target(target, self.settings.server)
        logger.info(
            "Набираю %s (INVITE sip:%s@%s)", target, number, self.settings.server
        )
        return self._phone.call(number)

    def answer(self, call: VoIPCall) -> bool:
        try:
            call.answer()
            return True
        except InvalidStateError as exc:
            logger.warning("Не удалось ответить: %s", exc)
            return False

    def deny(self, call: VoIPCall) -> bool:
        """Отклоняет входящий вызов (486 Busy Here)."""
        try:
            call.deny()
            logger.info("Вызов отклонён: %s", describe_caller(call))
            return True
        except InvalidStateError as exc:
            logger.warning("Не удалось отклонить: %s", exc)
            return False

    def hangup(self, call: VoIPCall) -> None:
        if getattr(call, "state", None) is CallState.ENDED:
            return
        try:
            call.hangup()
        except InvalidStateError:
            pass
        except Exception:  # noqa: BLE001
            logger.exception("Ошибка при завершении звонка")

    def wait_answered(self, call: VoIPCall, timeout: float = 20.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = getattr(call, "state", None)
            if state is CallState.ANSWERED:
                return True
            if state is CallState.ENDED:
                return False
            time.sleep(0.1)
        return getattr(call, "state", None) is CallState.ANSWERED

    def wait_ended(self, call: VoIPCall, duration: float = 0.0) -> str:
        """Ждёт завершения звонка.

        ``duration`` > 0 — ограничение по времени в секундах (0 = пока не положат
        трубку). Возвращает причину: ``ended`` / ``timeout``.
        """
        started = time.monotonic()
        while True:
            if getattr(call, "state", None) is CallState.ENDED:
                return "ended"
            if duration > 0 and time.monotonic() - started >= duration:
                return "timeout"
            time.sleep(0.1)


def describe_caller(call: VoIPCall) -> str:
    """Достаёт From/Call-ID из входящего INVITE для лога."""
    headers = {}
    request = getattr(call, "request", None)
    if request is not None:
        headers = getattr(request, "headers", {}) or {}
    return f"from={headers.get('From', '?')} call-id={getattr(call, 'call_id', '?')}"


__all__ = ["SipPhone", "normalize_target", "describe_caller"]
