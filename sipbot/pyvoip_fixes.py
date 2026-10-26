"""Патчи pyVoIP 1.6.8 для работы с реальными АТС (Asterisk и подобными).

Проверено на живой АТС (Asterisk 11.7). pyVoIP 1.6.8 в такой сети ломается
двумя связанными способами:

1. **Не отвечает на OPTIONS.** ``SIPMessage`` умеет разбирать только
   ``pyVoIP.SIPCompatibleMethods`` = INVITE/ACK/BYE/CANCEL, поэтому OPTIONS
   (keepalive/qualify от Asterisk) бросает ``SIPParseError`` и в recv-цикле
   молча выбрасывается. АТС не получает ответ и считает абонента недоступным
   (``qualify``), из-за чего входящие звонки могут не доставляться.
2. **Путает keepalive с ответом на REGISTER.** Если OPTIONS приходит между
   нашим REGISTER и ответом на него, внутренний ``__register`` вызывает
   ``SIPMessage(OPTIONS)``, получает ``SIPParseError``, считает регистрацию
   неуспешной, увеличивает ``registerFailures`` и запускает таймер с
   ``delay=0`` — в сеть уходит лишний REGISTER без авторизации. После трёх
   таких ошибок телефон останавливается со статусом FAILED.

Патчи (идемпотентны, применяются один раз):

* keepalive-методы добавляются в ``pyVoIP.SIPCompatibleMethods``, чтобы
  ``SIPMessage`` мог их разобрать;
* ``SIPClient.parse_message`` отвечает ``200 OK`` на сервисные запросы;
* ``SIPClient.recv`` и ``SIPClient.__register`` обслуживают «чужие» сообщения
  (OPTIONS и т.п.) вместо того, чтобы на них спотыкаться;
* ``RTPClient.trans`` ждёт абсолютный дедлайн пакета, а не ``time.sleep(20 мс)``
  (на Windows такой сон даёт 16/32 мс и рваный звук у абонента).

Публичный API: :func:`install`, :func:`handle_incoming`, :func:`answer_request`.
"""

from __future__ import annotations

import logging
import select
import time
import uuid
import warnings
from typing import Optional

import pyVoIP

# pyVoIP.SIP и pyVoIP.VoIP импортируют друг друга; если начать с SIP, получаем
# partially initialized module. Поэтому сначала инициализируем VoIP.
import pyVoIP.VoIP  # noqa: F401
from pyVoIP.SIP import (
    InvalidAccountInfoError,
    RetryRequiredError,
    SIPClient,
    SIPMessage,
    SIPMessageType,
    SIPStatus,
)
from pyVoIP.VoIP.status import PhoneStatus

# Важно: после pyVoIP.SIP. Если импортировать RTP первым, pyVoIP.SIP и
# pyVoIP.VoIP попадают в циклический импорт.
from pyVoIP.RTP import RTPClient  # noqa: E402

from .timing import now, wait_until

logger = logging.getLogger(__name__)

#: Запросы, на которые клиент отвечает 200 OK (keepalive и служебные).
ANSWERABLE_METHODS = (
    "OPTIONS",
    "NOTIFY",
    "INFO",
    "MESSAGE",
    "SUBSCRIBE",
    "REFER",
    "UPDATE",
)

_installed = False
_original_parse_message = None


def is_request(message: SIPMessage) -> bool:
    """True, если сообщение — запрос (а не ответ)."""
    return message.type == SIPMessageType.MESSAGE


def build_ok(client: SIPClient, message: SIPMessage) -> str:
    """Собирает 200 OK на запрос, подставляя tag в From, если его не было."""
    from_header = message.headers.get("From")
    if isinstance(from_header, dict) and not from_header.get("tag"):
        from_header["tag"] = uuid.uuid4().hex[:8]
    return client.gen_ok(message)


def answer_request(
    client: SIPClient,
    message: SIPMessage,
    addr: Optional[tuple] = None,
) -> bool:
    """Отвечает 200 OK на сервисный запрос.

    ``addr`` — адрес получателя; по умолчанию SIP-сервер (через него идут
    keepalive-запросы от АТС). Возвращает True, если запрос обработан.
    """
    method = (getattr(message, "method", "") or "").upper()
    if method not in ANSWERABLE_METHODS:
        return False
    response = build_ok(client, message)
    destination = addr or (client.server, client.port)
    client.out.sendto(response.encode("utf8"), destination)
    logger.debug(
        "Ответ 200 OK на %s (Call-ID %s) -> %s",
        method,
        message.headers.get("Call-ID"),
        destination,
    )
    return True


def handle_incoming(client: SIPClient, raw: bytes) -> Optional[SIPMessage]:
    """Разбирает входящий пакет и обслуживает сервисные запросы.

    Возвращает разобранное сообщение только для ответов (response), чтобы
    вызывающая сторона могла отфильтровать нужный ей ответ. Запросы
    (OPTIONS/INVITE/...) обслуживаются здесь же и возвращают None.
    """
    try:
        message = SIPMessage(raw)
    except Exception as exc:  # noqa: BLE001 - мусор в сети не должен ронять поток
        logger.debug("Пропущен неразбираемый SIP-пакет (%s байт): %s", len(raw), exc)
        return None

    if not is_request(message):
        return message

    if answer_request(client, message):
        return None
    # INVITE/BYE/CANCEL/ACK — штатная логика pyVoIP (созвон, отбой и т.д.)
    parse_message_safely(client, message)
    return None


def _read_register_response(client: SIPClient, deadline: float) -> SIPMessage:
    """Ждёт ответ именно на REGISTER, попутно отвечая на keepalive."""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Регистрация: АТС не ответила вовремя")
        ready = select.select([client.s], [], [], remaining)
        if not ready[0]:
            raise TimeoutError("Регистрация: АТС не ответила вовремя")
        try:
            raw = client.s.recv(8192)
        except BlockingIOError:
            continue
        except OSError as exc:
            raise TimeoutError(f"Регистрация: сокет закрыт ({exc})") from exc
        if not raw or raw == b"\x00\x00\x00\x00":
            continue

        message = handle_incoming(client, raw)
        if message is None:
            continue
        cseq = message.headers.get("CSeq")
        method = str(cseq.get("method", "")).upper() if isinstance(cseq, dict) else ""
        if method != "REGISTER":
            logger.debug("Во время регистрации пропущен ответ на %s", method or "?")
            continue
        if message.status == SIPStatus.TRYING:
            continue
        return message


def _register(client: SIPClient) -> bool:
    """Замена ``SIPClient.__register``: не спотыкается о keepalive."""
    client.phone._status = PhoneStatus.REGISTERING
    try:
        request = client.gen_first_response()
        client.out.sendto(request.encode("utf8"), (client.server, client.port))
        deadline = time.monotonic() + client.register_timeout
        response = _read_register_response(client, deadline)

        if response.status == SIPStatus.UNAUTHORIZED:
            auth_request = client.gen_register(response)
            client.out.sendto(auth_request.encode("utf8"), (client.server, client.port))
            deadline = time.monotonic() + client.register_timeout
            response = _read_register_response(client, deadline)
            if response.status == SIPStatus.UNAUTHORIZED:
                raise InvalidAccountInfoError(
                    "Неверные логин/пароль для SIP-сервера "
                    f"{client.server}:{client.myPort}"
                )

        if response.status == SIPStatus.INTERNAL_SERVER_ERROR:
            raise RetryRequiredError("Ответ SIP-сервера 500")

        if response.status != SIPStatus.OK:
            raise InvalidAccountInfoError(
                f"Регистрация отклонена: {response.status} "
                f"({client.server}:{client.myPort})"
            )

        logger.debug("Регистрация успешна (Call-ID %s)", response.headers.get("Call-ID"))
        return True
    finally:
        try:
            client.s.setblocking(True)
        except OSError:  # сокет мог быть уже закрыт
            pass


def parse_message_safely(client: SIPClient, message: SIPMessage) -> None:
    """Разбирает сообщение, не давая упасть потоку приёма.

    pyVoIP 1.6.8 роняет recv-поток, если приходит ответ на неизвестный Call-ID:
    ``VoIPPhone._callback_RESP_NotFound`` делает ``self.calls[call_id]`` без
    проверки (для REGISTER-ответов это норма). Пример из жизни: АТС отвечает
    ``404`` на несуществующий номер при исходящем звонке.
    """
    if not is_request(message):
        call_id = message.headers.get("Call-ID")
        calls = getattr(client.phone, "calls", None)
        if (
            message.status in (SIPStatus.NOT_FOUND, SIPStatus.SERVICE_UNAVAILABLE)
            and isinstance(calls, dict)
            and call_id not in calls
        ):
            logger.debug(
                "Ответ %s на неизвестный Call-ID %s пропущен", message.status, call_id
            )
            return
    try:
        client.parse_message(message)
    except Exception:  # noqa: BLE001 - поток приёма важнее одной ошибки
        logger.exception(
            "Ошибка обработки SIP-сообщения (%s)",
            getattr(message, "status", getattr(message, "method", "?")),
        )


def _parse_message(self: SIPClient, message: SIPMessage) -> None:
    """``SIPClient.parse_message`` с ответом на сервисные запросы."""
    if is_request(message) and answer_request(self, message):
        return
    _original_parse_message(self, message)  # type: ignore[misc]


def _recv(self: SIPClient) -> None:
    """``SIPClient.recv`` с обслуживанием OPTIONS/keepalive."""
    try:
        raw = self.s.recv(8192)
    except BlockingIOError:
        raise
    except OSError:
        return
    if not raw or raw == b"\x00\x00\x00\x00":
        return
    message = handle_incoming(self, raw)
    if message is not None:
        parse_message_safely(self, message)


def _trans(self: RTPClient) -> None:
    """``RTPClient.trans`` с точным тактом 20 мс на Windows.

    Оригинал спит ``time.sleep(delay)``; системный таймер просыпается через
    16 или 32 мс вместо 20. Пакеты уходят неравномерно, pyVoIP то опустошает
    очередь (и шлёт тишину), то копит её — абонент слышит провалы и «затухания».
    Здесь ждём абсолютный дедлайн, компенсируя время на саму отправку.
    """
    interval = (1 / self.preference.rate) * 160 / self.trans_delay_reduction
    next_at = now()
    while self.NSD:
        next_at += interval
        payload = self.encode_packet(self.pmout.read())
        packet = b"\x80"  # RFC 1889 V2 No Padding Extension or CC.
        packet += chr(int(self.preference)).encode("utf8")
        try:
            packet += self.outSequence.to_bytes(2, byteorder="big")
        except OverflowError:
            self.outSequence = 0
        try:
            packet += self.outTimestamp.to_bytes(4, byteorder="big")
        except OverflowError:
            self.outTimestamp = 0
        packet += self.outSSRC.to_bytes(4, byteorder="big")
        packet += payload

        try:
            self.sout.sendto(packet, (self.outIP, self.outPort))
        except OSError:
            warnings.warn(
                "RTP Packet failed to send!",
                RuntimeWarning,
                stacklevel=2,
            )

        self.outSequence += 1
        self.outTimestamp += len(payload)
        current = now()
        if next_at < current - 0.05:
            # Сильно отстали (планировщик ОС) — не копим долг, начинаем заново.
            next_at = current
        wait_until(next_at)


def _ack_request_uri(request: SIPMessage) -> str:
    """Request-URI для ACK: Contact из ответа, иначе адрес из To."""
    target = str(
        request.headers.get("Contact") or request.headers["To"]["raw"]
    ).strip()
    if "<" in target and ">" in target:
        return target.split("<", 1)[1].split(">", 1)[0]
    return target.strip("<").strip(">")


def _gen_ack(self: SIPClient, request: SIPMessage) -> str:
    """ACK с правильным диалоговым To-tag и Contact в Request-URI.

    В pyVoIP 1.6.8 ``gen_ack`` каждый раз генерирует новый To-tag. АТС не
    принимает такой ACK. Для ACK на 2xx формируется новая Via-branch
    (новая транзакция), для не-2xx сохраняется branch исходного INVITE.
    """
    call_id = request.headers["Call-ID"]
    local_tag = self.tagLibrary[call_id]
    to_header = request.headers["To"]["raw"]
    remote_tag = request.headers["To"].get("tag")
    if remote_tag and ";tag=" not in to_header.lower():
        to_header += f";tag={remote_tag}"

    ack = f"ACK {_ack_request_uri(request)} SIP/2.0\r\n"
    if getattr(request, "status", None) == SIPStatus.OK:
        branch = "z9hG4bK" + self.gen_call_id()[0:25]
        ack += f"Via: SIP/2.0/UDP {self.myIP}:{self.myPort};branch={branch};rport\r\n"
    else:
        ack += self._gen_response_via_header(request)
    ack += "Max-Forwards: 70\r\n"
    ack += f"To: {to_header}\r\n"
    ack += f"From: {request.headers['From']['raw']};tag={local_tag}\r\n"
    ack += f"Call-ID: {call_id}\r\n"
    ack += f"CSeq: {request.headers['CSeq']['check']} ACK\r\n"
    ack += f"User-Agent: pyVoIP {pyVoIP.__version__}\r\n"
    ack += "Content-Length: 0\r\n\r\n"
    return ack


def install() -> None:
    """Применяет патчи (повторный вызов ничего не делает)."""
    global _installed, _original_parse_message
    if _installed:
        return

    for method in ANSWERABLE_METHODS:
        if method not in pyVoIP.SIPCompatibleMethods:
            pyVoIP.SIPCompatibleMethods.append(method)

    _original_parse_message = SIPClient.parse_message
    SIPClient.parse_message = _parse_message  # type: ignore[method-assign]
    SIPClient.recv = _recv  # type: ignore[method-assign]
    SIPClient.gen_ack = _gen_ack  # type: ignore[method-assign]
    SIPClient._SIPClient__register = _register  # type: ignore[attr-defined]
    RTPClient.trans = _trans  # type: ignore[method-assign]

    _installed = True
    logger.debug(
        "Патчи pyVoIP применены: ответ 200 OK на %s, устойчивая регистрация, "
        "точный такт RTP",
        "/".join(ANSWERABLE_METHODS),
    )


def is_installed() -> bool:
    """True, если патчи уже применены."""
    return _installed


__all__ = [
    "ANSWERABLE_METHODS",
    "answer_request",
    "build_ok",
    "handle_incoming",
    "install",
    "is_installed",
    "is_request",
    "parse_message_safely",
]
