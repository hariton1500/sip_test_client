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
  (на Windows такой сон даёт 16/32 мс и рваный звук у абонента);
* **исходящий звук кодируется из 16-битного PCM** (см. :func:`_write`,
  :func:`_manager_read`, :func:`_encode_pcm16`). Штатный pyVoIP держит в
  буфере u8, а ``encode_pcmu`` затем делает точное u8 -> s16 -> μ-law;
  потери там нет, но приложение всё равно квантует звук в 8 бит
  (``tts.py``: ``s16_to_u8``). Патч готов принимать s16-кадры: как только
  боты/мост перестанут отдавать u8, SNR вырастет с ~28 до ~37 дБ.
  Сейчас выигрыша по SNR нет (проверено на кэше Gemini TTS);
* ``RTPClient.encode_packet`` для PCMA вызывает A-law, а не μ-law
  (в pyVoIP 1.6.8 там ошибка: ``elif self.preference == PayloadType.PCMA:
  return self.encode_pcmu(payload)`` — при согласовании PCMA абонент слышит
  сплошной шум);
* при ``SIPBOT_TX_WAV=<путь>`` исходящий поток пишется в WAV 8 кГц/16 бит —
  это ровно то, что уходит в сеть (диагностика: клиент или транскодинг
  оператора портит звук);
* передатчик сам выбирает G.711 из согласованных кодеков (а не первый payload
  из SDP) и переживает сбой отдельного пакета: иначе ответ АТС с G.729 первым
  убивал ``RTPClient.trans`` и в линию не уходило ни одного кадра;
* ответы АТС, которые pyVoIP молча пропускает (403/407/486/…), пишутся в лог:
  раньше отказ выглядел как «звонок не отвечает»;
* ``SIPClient.invite`` возвращается на первый же ответ для своего Call-ID:
  исходный цикл ждал только ``100``/``180``/``401`` и на ``183 Session
  Progress`` мог остаться в ``recv()`` — исходящий звонок соединялся, но
  приложение не запускало мост (тишина в трубке).

Публичный API: :func:`install`, :func:`handle_incoming`, :func:`answer_request`.
"""

from __future__ import annotations

import logging
import os
import re
import select
import threading
import time
import uuid
import warnings
import wave
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
from pyVoIP.RTP import (  # noqa: E402
    PayloadType,
    RTPClient,
    RTPPacketManager,
    RTPParseError,
)

from .audio_utils import s16_to_alaw, s16_to_ulaw, u8_to_s16
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


#: Ответы, которые pyVoIP 1.6.8 разбирает сам (см. ``SIPClient.parse_message``).
HANDLED_RESPONSES = (
    SIPStatus.OK,
    SIPStatus.NOT_FOUND,
    SIPStatus.SERVICE_UNAVAILABLE,
    SIPStatus.TRYING,
    SIPStatus.RINGING,
)


def _cseq_hint(message: SIPMessage) -> str:
    """«2 INVITE» из заголовка CSeq (или пустая строка, если CSeq нет)."""
    cseq = message.headers.get("CSeq") or {}
    if isinstance(cseq, dict):
        return f"{cseq.get('check')} {cseq.get('method')}".strip()
    return str(cseq)


def _via_hint(message: SIPMessage) -> str:
    """branch верхнего Via: по нему видно, ретрансмиссия это или новый запрос."""
    via = message.headers.get("Via") or []
    first = via[0] if isinstance(via, list) and via else via
    if isinstance(first, dict):
        return str(first.get("branch", ""))
    return ""


def _parse_message(self: SIPClient, message: SIPMessage) -> None:
    """``SIPClient.parse_message`` с ответом на сервисные запросы.

    Дополнительно логирует ответы, которые pyVoIP молча пропускает. Без этого
    отказ АТС (например, ``407 Proxy Authentication Required`` на INVITE)
    выглядит как «звонок не отвечает»: pyVoIP печатает только
    ``TODO: Add 500 Error on Receiving SIP Response`` без статуса.
    """
    if is_request(message):
        logger.debug(
            "SIP: <- %s (CSeq %s, Via %s, Call-ID %s)",
            message.method,
            _cseq_hint(message),
            _via_hint(message),
            message.headers.get("Call-ID"),
        )
        if answer_request(self, message):
            return
    else:
        logger.debug(
            "SIP: <- %s (CSeq %s, Via %s, Call-ID %s)",
            message.status,
            _cseq_hint(message),
            _via_hint(message),
            message.headers.get("Call-ID"),
        )
        if message.status == SIPStatus.OK and _cseq_hint(message).endswith("INVITE"):
            try:
                ack_uri = _ack_request_uri(message)
            except Exception:  # noqa: BLE001 - в логах важнее не упасть
                ack_uri = "?"
            logger.debug(
                "SIP: 200 OK на INVITE: ACK Request-URI %s, Contact %s, "
                "Record-Route %s",
                ack_uri,
                message.headers.get("Contact"),
                message.headers.get("Record-Route"),
            )
            if message.headers.get("Call-ID") in self.tagLibrary:
                _ack_invite_response(self, message)
        if message.status not in HANDLED_RESPONSES:
            level = logging.DEBUG if int(message.status) < 200 else logging.WARNING
            logger.log(
                level,
                "SIP: ответ %s не обрабатывается pyVoIP (Call-ID %s)",
                message.status,
                message.headers.get("Call-ID"),
            )
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


#: Сэмплов в одном RTP-пакете (20 мс при 8 кГц).
FRAME_SAMPLES = 160
#: Байт в 16-битном кадре 20 мс.
PCM16_FRAME_BYTES = FRAME_SAMPLES * 2
#: Тишина во внутреннем формате pyVoIP (u8, 0x80).
U8_SILENCE = b"\x80"

_TX_TAP_UNSET = object()


def _manager_read(self: RTPPacketManager, length: int = 160) -> bytes:
    """``RTPPacketManager.read`` с тишиной того формата, что лежит в буфере.

    Оригинал всегда добивает недостачу байтом ``0x80``. Для исходящего
    буфера это уже 16-битный PCM, и ``0x80`` читается как ``-32640`` —
    щелчок на каждом недосыле. Поэтому ``RTPClient.trans`` выставляет
    ``pmout.silence = b"\\x00"``, а входящий буфер остаётся на ``0x80``.
    """
    while self.rebuilding:
        time.sleep(0.01)
    with self.bufferLock:
        packet = self.buffer.read(length)
        if len(packet) < length:
            silence = getattr(self, "silence", U8_SILENCE)
            missing = length - len(packet)
            packet += (silence * (missing // len(silence) + 1))[:missing]
    return packet


def _write(self: RTPClient, data: bytes) -> None:
    """``RTPClient.write``: u8 от приложения -> 16 бит в исходящий буфер.

    Штатный pyVoIP держит в ``pmout`` 8-битные сэмплы и кодирует их через
    ``lin2ulaw(..., 1)``, что эквивалентно точному u8 -> s16 -> μ-law.
    Мы храним s16le: это позволяет кодировать G.711 прямо из 16 бит, без
    потери точности на промежуточном u8, и делает возможной отправку
    s16-кадров без изменения остального тракта.
    """
    pcm16 = u8_to_s16(data)
    self.pmout.write(self.outOffset, pcm16)
    self.outOffset += len(pcm16)


def _encode_pcm16(pcm16: bytes, preference: PayloadType) -> bytes:
    """Кодирует 16-битный PCM в payload согласованного кодека."""
    if preference == PayloadType.PCMU:
        return s16_to_ulaw(pcm16)
    if preference == PayloadType.PCMA:
        return s16_to_alaw(pcm16)
    raise RTPParseError(f"Отправка кодека {preference} не поддерживается")


def _encode_packet(self: RTPClient, payload: bytes) -> bytes:
    """``RTPClient.encode_packet`` по согласованному кодеку.

    В pyVoIP 1.6.8 ветка PCMA ошибочно вызывает ``encode_pcmu``: при
    согласовании PCMA абонент получает μ-law под меткой A-law (сплошной шум).
    """
    return _encode_pcm16(payload, self.preference)


#: Кодеки, звук в которые мы умеем кодировать (G.711).
AUDIO_PAYLOADS = (PayloadType.PCMU, PayloadType.PCMA)


def select_audio_codec(client: RTPClient) -> Optional[PayloadType]:
    """Первый поддерживаемый аудиокодек из согласованных с АТС.

    ``RTPClient`` из pyVoIP 1.6.8 берёт первый payload type из ответа SDP как
    есть. Если АТС перечислила первой не G.711 (например, G.729), передатчик
    падает на кодировании первого же пакета: ``RTPClient.trans`` умирает, в
    линию не уходит ни одного кадра, и абонент слышит тишину (оператор при
    этом через несколько секунд сбрасывает звонок).
    """
    for payload in client.assoc.values():
        if payload in AUDIO_PAYLOADS:
            return payload
    return None


class _TxTap:
    """Пишет исходящий s16-поток в WAV — диагностика «что уходит в сеть»."""

    def __init__(self, path: str, rate: int) -> None:
        self.path = path
        self.frames = 0
        self._wf: Optional[wave.Wave_write] = None
        self._lock = threading.Lock()
        try:
            wf = wave.open(path, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
        except (OSError, wave.Error):
            logger.warning("Не удалось открыть TX-дамп %s", path, exc_info=True)
            return
        self._wf = wf

    @property
    def enabled(self) -> bool:
        return self._wf is not None

    def write(self, pcm16: bytes) -> None:
        if self._wf is None:
            return
        with self._lock:
            self._wf.writeframes(pcm16)
            self.frames += 1

    def close(self) -> None:
        wf, self._wf = self._wf, None
        if wf is None:
            return
        try:
            wf.close()
        except (OSError, wave.Error):
            logger.warning("Ошибка закрытия TX-дампа %s", self.path, exc_info=True)
            return
        logger.info("TX-дамп записан: %s (%d кадров)", self.path, self.frames)


def _tx_tap_for(client: RTPClient) -> Optional[_TxTap]:
    """Возвращает TX-дамп клиента, создавая его при первом пакете.

    Путь берётся из ``SIPBOT_TX_WAV``; если переменная не задана, дамп
    отключён и повторные проверки не выполняются.
    """
    tap = getattr(client, "_sipbot_tx_tap", _TX_TAP_UNSET)
    if tap is not _TX_TAP_UNSET:
        return tap  # type: ignore[return-value]
    path = os.environ.get("SIPBOT_TX_WAV", "").strip()
    writer: Optional[_TxTap] = None
    if path:
        writer = _TxTap(path, client.preference.rate)
        if writer.enabled:
            logger.info("TX-дамп исходящего аудио: %s", path)
        else:
            writer = None
    client._sipbot_tx_tap = writer  # type: ignore[attr-defined]
    return writer


def _close_tx_tap(client: RTPClient) -> None:
    tap = getattr(client, "_sipbot_tx_tap", None)
    if isinstance(tap, _TxTap):
        tap.close()


def _log_codec(client: RTPClient) -> None:
    logger.info(
        "RTP: кодек %s (PT %s, %s Гц), исходящий поток %s:%s -> %s:%s",
        client.preference,
        int(client.preference),
        client.preference.rate,
        client.inIP,
        client.inPort,
        client.outIP,
        client.outPort,
    )
    logger.debug(
        "RTP: согласованные типы %s",
        {pt: str(payload) for pt, payload in client.assoc.items()},
    )


def _transmit_once(self: RTPClient) -> bytes:
    """Формирует и отправляет один 20-мс RTP-пакет, возвращает payload.

    Вынесено из цикла :func:`_trans`, чтобы тесты могли проверить один
    пакет без таймеров и сокета.
    """
    self.pmout.silence = b"\x00"  # 16-битная тишина вместо 0x80
    pcm16 = self.pmout.read(PCM16_FRAME_BYTES)
    tap = _tx_tap_for(self)
    if tap is not None:
        tap.write(pcm16)
    payload = _encode_pcm16(pcm16, self.preference)
    if not getattr(self, "_sipbot_tx_started", False):
        self._sipbot_tx_started = True  # type: ignore[attr-defined]
        logger.info(
            "RTP: первый пакет в линию -> %s:%s (%s, payload %d байт)",
            self.outIP,
            self.outPort,
            self.preference,
            len(payload),
        )

    packet = b"\x80"  # RFC 1889 V2 No Padding Extension or CC.
    packet += chr(int(self.preference)).encode("utf8")
    packet += (self.outSequence % 0x10000).to_bytes(2, byteorder="big")
    packet += (self.outTimestamp % 0x100000000).to_bytes(4, byteorder="big")
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

    self.outSequence = (self.outSequence + 1) % 0x10000
    self.outTimestamp = (self.outTimestamp + FRAME_SAMPLES) % 0x100000000
    return payload



def _trans(self: RTPClient) -> None:
    """``RTPClient.trans`` с точным тактом 20 мс на Windows.

    Оригинал спит ``time.sleep(delay)``; системный таймер просыпается через
    16 или 32 мс вместо 20. Пакеты уходят неравномерно, pyVoIP то опустошает
    очередь (и шлёт тишину), то копит её — абонент слышит провалы и «затухания».
    Здесь ждём абсолютный дедлайн.

    Дополнительно: в исходящий буфер кладётся 16-битный PCM (см. :func:`_write`),
    поэтому читаем 320 байт и кодируем G.711 сразу из 16 бит.
    """
    codec = select_audio_codec(self)
    if codec is None:
        logger.error(
            "RTP: в ответе АТС нет PCMU/PCMA (%s) — отправка звука невозможна",
            {pt: str(payload) for pt, payload in self.assoc.items()},
        )
        _close_tx_tap(self)
        return
    if codec is not self.preference:
        logger.warning(
            "RTP: pyVoIP выбрал %s, переключаюсь на %s", self.preference, codec
        )
        self.preference = codec

    _log_codec(self)
    interval = (1 / self.preference.rate) * FRAME_SAMPLES / self.trans_delay_reduction
    next_at = now()
    failures = 0
    try:
        while self.NSD:
            next_at += interval
            try:
                _transmit_once(self)
                failures = 0
            except Exception:  # noqa: BLE001 - один сбой не должен глушить линию
                failures += 1
                # Не засоряем лог: первый сбой с трейсбеком, дальше каждый 50-й.
                if failures == 1 or failures % 50 == 0:
                    logger.exception(
                        "RTP: не удалось отправить пакет (подряд %d)", failures
                    )
                if failures >= 200:
                    logger.error(
                        "RTP: отправка остановлена после %d ошибок подряд", failures
                    )
                    return
            current = now()
            if next_at < current - 0.05:
                # Сильно отстали (планировщик ОС) — не копим долг, начинаем заново.
                next_at = current
            wait_until(next_at)
    finally:
        _close_tx_tap(self)


def _ack_request_uri(request: SIPMessage) -> str:
    """Request-URI для ACK: Contact из ответа, иначе адрес из To."""
    target = str(
        request.headers.get("Contact") or request.headers["To"]["raw"]
    ).strip()
    if "<" in target and ">" in target:
        return target.split("<", 1)[1].split(">", 1)[0]
    return target.strip("<").strip(">")


def _ack_destination(client: SIPClient, request: SIPMessage) -> tuple:
    """Куда слать ACK: ``host:port`` из Contact ответа, иначе адрес сервера.

    RFC 3261 (13.2.2.4) требует отправлять ACK на 2xx по URI из ``Contact``.
    Сервер и Contact обычно совпадают, но при NAT/прокси адрес из Contact
    надёжнее: именно его АТС записала как remote target диалога.
    """
    try:
        uri = _ack_request_uri(request)
        host_port = uri.split("@", 1)[1].split(";", 1)[0]
        host, _, port = host_port.rpartition(":")
        if host and port.isdigit():
            return (host, int(port))
    except Exception:  # noqa: BLE001 - разбор Contact не должен рвать звонок
        logger.debug(
            "Не удалось разобрать Contact для ACK: %r",
            request.headers.get("Contact"),
        )
    return (client.server, client.port)


#: Сколько раз АТС повторила 200 OK по каждому Call-ID (ключ -> счётчик).
_ACK_COUNTS: Dict[str, int] = {}
#: Предел размера ``_ACK_COUNTS``, чтобы словарь не рос на длинной сессии.
_ACK_COUNTS_MAX = 64


def _ack_invite_response(client: SIPClient, request: SIPMessage) -> None:
    """Страховочный ACK на повторный ``200 OK``.

    pyVoIP подтверждает каждый 200 OK, но некоторая АТС (Asterisk 11.x)
    принимает ACK на 2xx только в транзакции INVITE: со ``branch`` исходного
    INVITE. Если 200 OK повторился — первый ACK не принят, поэтому отправляем
    второй вариант ACK (branch INVITE) прямо на адрес из ``Contact``.
    Повторные ACK для 2xx по RFC 3261 допустимы.
    """
    call_id = str(request.headers.get("Call-ID") or "")
    count = _ACK_COUNTS.get(call_id, 0) + 1
    if len(_ACK_COUNTS) > _ACK_COUNTS_MAX:
        _ACK_COUNTS.clear()
    _ACK_COUNTS[call_id] = count
    if count == 1:
        return
    try:
        ack = _gen_ack(client, request, same_transaction=True)
    except Exception:  # noqa: BLE001 - ACK не должен рвать обработку ответа
        logger.exception("Не удалось построить страховочный ACK на 200 OK")
        return
    destination = _ack_destination(client, request)
    try:
        client.out.sendto(ack.encode("utf8"), destination)
    except OSError:
        logger.exception("Страховочный ACK не отправлен (%s)", destination)
        return
    logger.info(
        "ACK на повтор 200 OK #%d -> %s:%s (branch исходного INVITE)",
        count - 1,
        destination[0],
        destination[1],
    )


def _gen_ack(
    self: SIPClient, request: SIPMessage, same_transaction: bool = False
) -> str:
    """ACK с правильным диалоговым To-tag и Contact в Request-URI.

    В pyVoIP 1.6.8 ``gen_ack`` каждый раз генерирует новый To-tag. АТС не
    принимает такой ACK. Для ACK на 2xx формируется новая Via-branch
    (новая транзакция), для не-2xx сохраняется branch исходного INVITE.

    ``same_transaction`` принудительно оставляет branch исходного INVITE:
    так АТС, требующая ACK в транзакции INVITE, принимает подтверждение.
    """
    call_id = request.headers["Call-ID"]
    local_tag = self.tagLibrary[call_id]
    to_header = request.headers["To"]["raw"]
    remote_tag = request.headers["To"].get("tag")
    if remote_tag and ";tag=" not in to_header.lower():
        to_header += f";tag={remote_tag}"

    ack = f"ACK {_ack_request_uri(request)} SIP/2.0\r\n"
    is_ok = getattr(request, "status", None) == SIPStatus.OK
    if is_ok and not same_transaction:
        branch = "z9hG4bK" + self.gen_call_id()[0:25]
        ack += f"Via: SIP/2.0/UDP {self.myIP}:{self.myPort};branch={branch};rport\r\n"
    elif is_ok:
        branch = _via_hint(request) or "z9hG4bK" + self.gen_call_id()[0:25]
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
    logger.debug(
        "ACK -> %s (To %s, CSeq %s ACK)",
        ack.split("\r\n", 1)[0],
        to_header,
        request.headers["CSeq"]["check"],
    )
    return ack


#: Сколько ждать первый ответ на INVITE, прежде чем отдать управление дальше.
#: Ответы приходят за доли секунды; остальное (200 OK, BYE) ловит recv-поток.
INVITE_FIRST_RESPONSE_TIMEOUT = 5.0


_CHALLENGE_RE = re.compile(r'(realm|nonce)\s*=\s*"?([^",]+)"?')


def _ensure_proxy_challenge(response: SIPMessage) -> None:
    """Достаёт realm/nonce из ``Proxy-Authenticate`` (pyVoIP их не разбирает)."""
    if response.authentication.get("realm") and response.authentication.get(
        "nonce"
    ):
        return
    raw = response.headers.get("Proxy-Authenticate")
    if not isinstance(raw, str):
        return
    parsed = {key: value for key, value in _CHALLENGE_RE.findall(raw)}
    if parsed:
        response.authentication.update(parsed)


def _authorize_invite(
    self: SIPClient,
    response: SIPMessage,
    number: str,
    sess_id: int,
    ms: Dict[int, Dict[int, PayloadType]],
    sendtype,
    call_id: str,
) -> str:
    """Отвечает на ``401``/``407`` авторизованным INVITE (как pyVoIP для REGISTER).

    Два обязательных условия, без которых АТС (Asterisk) молча игнорирует
    повтор: не-2xx финальный ответ подтверждается ``ACK``, а сам INVITE должен
    быть **новой транзакцией** — свежий ``branch`` и следующий ``CSeq``. Если
    переслать тот же текст с добавленным заголовком, АТС считает его
    ретрансмиссией, повторяет ``401`` (pyVoIP такой ответ «обрабатывает» без
    записи в лог) и звонок висит до таймаута.

    Возвращает текст отправленного INVITE: вызывающий код должен вернуть именно
    его, иначе ``ACK`` на ``200 OK`` уйдёт с прежним ``CSeq``.
    """
    ack = self.gen_ack(response)
    self.out.sendto(ack.encode("utf8"), (self.server, self.port))
    try:
        _ensure_proxy_challenge(response)
        authhash = self.gen_authorization(response)
        nonce = response.authentication["nonce"]
        realm = response.authentication["realm"]
    except Exception:  # noqa: BLE001 - без авторизации звонок всё равно не выйдет
        logger.exception("Не удалось построить авторизацию INVITE")
        return ""

    header = (
        "Proxy-Authorization"
        if response.status == SIPStatus.PROXY_AUTHENTICATION_REQUIRED
        else "Authorization"
    )
    auth = (
        f'{header}: Digest username="{self.username}",realm="{realm}",'
        f'nonce="{nonce}",uri="sip:{self.server};transport=UDP",'
        f'response="{authhash.decode("utf8")}",algorithm=MD5\r\n'
    )
    invite = self.gen_invite(
        number, str(sess_id), ms, sendtype, self.gen_branch(), call_id
    )
    authorized = invite.replace(
        "\r\nContent-Length", f"\r\n{auth}Content-Length", 1
    )
    self.out.sendto(authorized.encode("utf8"), (self.server, self.port))
    _, _, cseq = authorized.partition("CSeq: ")
    logger.info(
        "Повторный INVITE с авторизацией (%s, CSeq %s)",
        header,
        cseq.split("\r\n", 1)[0],
    )
    return authorized


def _invite(
    self: SIPClient,
    number: str,
    ms: Dict[int, Dict[int, PayloadType]],
    sendtype,
) -> tuple:
    """``SIPClient.invite``, который не зависает на ``183 Session Progress``.

    В pyVoIP 1.6.8 цикл ждёт только ``100``/``180``/``401``: если АТС сначала
    присылает ``183`` (early media), ``call()`` остаётся в ``recv()`` навсегда
    и вызывающий код не успевает запустить мост — абонент слышит тишину при
    соединённом звонке. Возвращаемся на первый же ответ для нашего Call-ID:
    остальные (200 OK, BYE) обработает поток приёма.
    """
    branch = "z9hG4bK" + self.gen_call_id()[0:25]
    call_id = self.gen_call_id()
    sess_id = self.sessID.next()
    invite = self.gen_invite(number, str(sess_id), ms, sendtype, branch, call_id)
    deadline = time.monotonic() + INVITE_FIRST_RESPONSE_TIMEOUT
    with self.recvLock:
        self.out.sendto(invite.encode("utf8"), (self.server, self.port))
        logger.debug("INVITE отправлен (Call-ID %s)", call_id)
        while self.NSD and time.monotonic() < deadline:
            try:
                ready = select.select([self.s], [], [], 0.2)[0]
            except (OSError, ValueError):
                break
            if not ready:
                continue
            try:
                raw = self.s.recv(8192)
            except BlockingIOError:
                continue
            except OSError:
                break
            if not raw or raw == b"\x00\x00\x00\x00":
                continue
            try:
                response = SIPMessage(raw)
            except Exception:  # noqa: BLE001 - мусор в сети не должен рвать звонок
                logger.debug("Пропущен неразбираемый ответ на INVITE")
                continue
            if response.headers.get("Call-ID") != call_id:
                parse_message_safely(self, response)
                continue
            if response.status in (
                SIPStatus.UNAUTHORIZED,
                SIPStatus.PROXY_AUTHENTICATION_REQUIRED,
            ):
                authorized = _authorize_invite(
                    self, response, number, sess_id, ms, sendtype, call_id
                )
                if authorized:
                    invite = authorized
                break
            parse_message_safely(self, response)
            break
        else:
            logger.debug(
                "Ответ на INVITE не пришёл за %.1f с, ждём в фоне",
                INVITE_FIRST_RESPONSE_TIMEOUT,
            )
    return SIPMessage(invite.encode("utf8")), call_id, sess_id


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
    SIPClient.invite = _invite  # type: ignore[method-assign]
    SIPClient._SIPClient__register = _register  # type: ignore[attr-defined]
    RTPClient.trans = _trans  # type: ignore[method-assign]
    RTPClient.write = _write  # type: ignore[method-assign]
    RTPClient.encode_packet = _encode_packet  # type: ignore[method-assign]
    RTPPacketManager.read = _manager_read  # type: ignore[method-assign]

    _installed = True
    logger.debug(
        "Патчи pyVoIP применены: ответ 200 OK на %s, устойчивая регистрация, "
        "INVITE без зависания на 183, точный такт RTP, 16-битный исходящий "
        "тракт, выбор G.711 и PCMA/A-law",
        "/".join(ANSWERABLE_METHODS),
    )


def is_installed() -> bool:
    """True, если патчи уже применены."""
    return _installed


__all__ = [
    "ANSWERABLE_METHODS",
    "AUDIO_PAYLOADS",
    "HANDLED_RESPONSES",
    "INVITE_FIRST_RESPONSE_TIMEOUT",
    "answer_request",
    "build_ok",
    "handle_incoming",
    "install",
    "is_installed",
    "is_request",
    "parse_message_safely",
    "select_audio_codec",
]
