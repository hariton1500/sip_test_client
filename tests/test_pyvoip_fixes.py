"""Тесты патчей pyVoIP: ответ на OPTIONS и устойчивая регистрация.

Всё оффлайн: вместо UDP-сокета — фейк с заранее заготовленными пакетами,
вместо select — заглушка, которая «готова», только если в очереди есть данные.
Сырые сообщения взяты из реального обмена с Asterisk 11.7.
"""

from types import SimpleNamespace

import logging
import math
import warnings
import wave

import numpy as np
import pyVoIP
import pytest

# pyVoIP.VoIP импортирует SIP и наоборот; без этого файл не собирается отдельно.
import pyVoIP.VoIP  # noqa: F401
from pyVoIP.RTP import PayloadType, TransmitType  # noqa: E402
from pyVoIP.SIP import (
    InvalidAccountInfoError,
    RetryRequiredError,
    SIPClient,
    SIPMessage,
    SIPStatus,
)

from sipbot import audio_utils, pyvoip_fixes

SESSION_IP = "192.168.1.5"
SERVER = "sip.example.com"


class FakeSocket:
    """Минимальная замена UDP-сокета для сокетного кода pyVoIP."""

    def __init__(self, incoming):
        self.incoming = list(incoming)
        self.sent = []
        self.blocking = True

    def recv(self, size=8192):
        if not self.incoming:
            raise BlockingIOError("нет данных")
        return self.incoming.pop(0)

    def sendto(self, data, addr):
        self.sent.append((bytes(data).decode("utf8"), addr))
        return len(data)

    def setblocking(self, flag):
        self.blocking = flag

    def close(self):
        pass


@pytest.fixture(autouse=True, scope="module")
def installed():
    pyvoip_fixes.install()
    return pyvoip_fixes.is_installed()


@pytest.fixture(autouse=True)
def fake_select(monkeypatch):
    """select готов, только если в фейковом сокете есть данные."""

    def _select(rlist, wlist, xlist, timeout=None):
        ready = [sock for sock in rlist if getattr(sock, "incoming", None)]
        return (ready, wlist, xlist)

    monkeypatch.setattr(pyvoip_fixes.select, "select", _select)


def make_client(incoming=()):
    phone = SimpleNamespace(_status=None, calls={})
    client = SIPClient(
        SERVER,
        5060,
        "503",
        "secret",
        phone,  # type: ignore[arg-type]
        myIP=SESSION_IP,
        myPort=5060,
    )
    sock = FakeSocket(incoming)
    client.s = sock
    client.out = sock
    # Как у запущенного клиента: без этого циклы ждут NSD и сразу выходят.
    client.NSD = True
    return client, sock


def options_raw(
    call_id: str = "opt-1@91.195.230.4",
    cseq: str = "102 OPTIONS",
    from_tag: str = "as5cbeb2cf",
) -> bytes:
    return (
        b"OPTIONS sip:503@192.168.1.5:5060;transport=UDP SIP/2.0\r\n"
        b"Via: SIP/2.0/UDP 91.195.230.4:5060;branch=z9hG4bK204d0071;rport\r\n"
        b"Max-Forwards: 70\r\n"
        b'From: "asterisk" <sip:asterisk@91.195.230.4>;tag='
        + from_tag.encode()
        + b"\r\n"
        b"To: <sip:503@192.168.1.5:5060;transport=UDP>\r\n"
        b"Contact: <sip:asterisk@91.195.230.4:5060>\r\n"
        b"Call-ID: " + call_id.encode() + b"\r\n"
        b"CSeq: " + cseq.encode() + b"\r\n"
        b"User-Agent: Asterisk PBX 11.7.0\r\n"
        b"Content-Length: 0\r\n\r\n"
    )


def register_401_raw(call_id: str = "reg-1@192.168.1.5:5060") -> bytes:
    return (
        b"SIP/2.0 401 Unauthorized\r\n"
        b"Via: SIP/2.0/UDP 192.168.1.5:5060;branch=z9hG4bK1;"
        b"received=192.168.1.5;rport=5060\r\n"
        b'From: "503" <sip:503@sip.example.com>;tag=a8c9c62d\r\n'
        b'To: "503" <sip:503@sip.example.com>;tag=as06d26d3f\r\n'
        b"Call-ID: " + call_id.encode() + b"\r\n"
        b"CSeq: 1 REGISTER\r\n"
        b"Server: Asterisk PBX 11.7.0\r\n"
        b'WWW-Authenticate: Digest algorithm=MD5, realm="asterisk", '
        b'nonce="25cd6ada"\r\n'
        b"Content-Length: 0\r\n\r\n"
    )


def register_200_raw(call_id: str = "reg-1@192.168.1.5:5060") -> bytes:
    return (
        b"SIP/2.0 200 OK\r\n"
        b"Via: SIP/2.0/UDP 192.168.1.5:5060;branch=z9hG4bK2;"
        b"received=192.168.1.5;rport=5060\r\n"
        b'From: "503" <sip:503@sip.example.com>;tag=a8c9c62d\r\n'
        b'To: "503" <sip:503@sip.example.com>;tag=as06d26d3f\r\n'
        b"Call-ID: " + call_id.encode() + b"\r\n"
        b"CSeq: 2 REGISTER\r\n"
        b"Server: Asterisk PBX 11.7.0\r\n"
        b"Expires: 120\r\n"
        b"Contact: <sip:503@192.168.1.5:5060;transport=UDP>;expires=120\r\n"
        b"Content-Length: 0\r\n\r\n"
    )


def register_status_raw(status_line: bytes, cseq: bytes = b"2 REGISTER") -> bytes:
    return (
        b"SIP/2.0 " + status_line + b"\r\n"
        b"Via: SIP/2.0/UDP 192.168.1.5:5060;branch=z9hG4bK2;rport=5060\r\n"
        b'From: "503" <sip:503@sip.example.com>;tag=a8c9c62d\r\n'
        b'To: "503" <sip:503@sip.example.com>;tag=as06d26d3f\r\n'
        b"Call-ID: reg-1@192.168.1.5:5060\r\n"
        b"CSeq: " + cseq + b"\r\n"
        b"Content-Length: 0\r\n\r\n"
    )


# ---------------------------------------------------------------------------
# Разбор и ответ на OPTIONS
# ---------------------------------------------------------------------------


def test_options_is_parsed_after_install():
    message = SIPMessage(options_raw())
    assert message.method == "OPTIONS"
    assert pyvoip_fixes.is_request(message)


def test_answer_options_sends_200_ok_with_same_call_id():
    client, sock = make_client()
    message = SIPMessage(options_raw())

    assert pyvoip_fixes.answer_request(client, message) is True
    assert len(sock.sent) == 1
    response, addr = sock.sent[0]
    assert response.startswith("SIP/2.0 200 OK")
    assert "Call-ID: opt-1@91.195.230.4" in response
    assert "CSeq: 102 OPTIONS" in response
    assert "Via: SIP/2.0/UDP 91.195.230.4:5060" in response
    assert addr == (SERVER, 5060)


def test_answer_request_ignores_invite():
    client, sock = make_client()
    message = SIPMessage(
        b"INVITE sip:503@192.168.1.5 SIP/2.0\r\n"
        b"Via: SIP/2.0/UDP 91.195.230.4:5060;branch=z9hG4bK3;rport\r\n"
        b'From: "100" <sip:100@91.195.230.4>;tag=abc\r\n'
        b"To: <sip:503@192.168.1.5>\r\n"
        b"Call-ID: call-1\r\n"
        b"CSeq: 1 INVITE\r\n"
        b"Contact: <sip:100@91.195.230.4:5060>\r\n"
        b"Content-Length: 0\r\n\r\n"
    )
    assert pyvoip_fixes.answer_request(client, message) is False
    assert sock.sent == []


def test_build_ok_adds_tag_when_from_has_none():
    client, _ = make_client()
    message = SIPMessage(options_raw(from_tag=""))
    assert message.headers["From"]["tag"] == ""

    response = pyvoip_fixes.build_ok(client, message)
    assert ";tag=" in response
    assert ";tag=\r\n" not in response


def test_handle_incoming_answers_options_and_returns_none():
    client, sock = make_client()
    assert pyvoip_fixes.handle_incoming(client, options_raw()) is None
    assert len(sock.sent) == 1
    assert sock.sent[0][0].startswith("SIP/2.0 200 OK")


def test_handle_incoming_returns_response():
    client, _ = make_client()
    message = pyvoip_fixes.handle_incoming(client, register_200_raw())
    assert message is not None
    assert message.status == SIPStatus.OK


def test_handle_incoming_ignores_garbage():
    client, sock = make_client()
    assert pyvoip_fixes.handle_incoming(client, b"\x00\x01garbage") is None
    assert sock.sent == []


def invite_200_raw() -> bytes:
    """200 OK на исходящий INVITE из реального обмена (теги сохранены)."""
    return (
        b"SIP/2.0 200 OK\r\n"
        b"Via: SIP/2.0/UDP 192.168.1.5:5060;branch=z9hG4bKold;"
        b"received=192.168.1.5;rport=5060\r\n"
        b"From: <sip:503@192.168.1.5>;tag=localtag\r\n"
        b"To: <sip:+79780481885@sip.example.com>;tag=as53dd332c\r\n"
        b"Call-ID: call-ack-1\r\n"
        b"CSeq: 2 INVITE\r\n"
        b"Contact: <sip:+79780481885@91.195.230.4:5060>\r\n"
        b"Content-Length: 0\r\n\r\n"
    )


def test_ack_keeps_dialog_tag_and_uses_contact():
    """ACK на 200 OK должен подтверждать диалог, а не создавать новый To-tag."""
    client, _ = make_client()
    client.tagLibrary["call-ack-1"] = "localtag"

    ack = client.gen_ack(SIPMessage(invite_200_raw()))

    assert ack.startswith("ACK sip:+79780481885@91.195.230.4:5060 SIP/2.0\r\n")
    assert "To: <sip:+79780481885@sip.example.com>;tag=as53dd332c\r\n" in ack
    assert "From: <sip:503@192.168.1.5>;tag=localtag\r\n" in ack
    assert "CSeq: 2 ACK\r\n" in ack
    assert "branch=z9hG4bKold" not in ack  # для 2xx — новая транзакция


def test_ack_for_non_2xx_keeps_transaction_branch():
    client, _ = make_client()
    call_id = "reg-1@192.168.1.5:5060"
    client.tagLibrary[call_id] = "localtag"

    ack = client.gen_ack(SIPMessage(register_401_raw()))

    assert ack.startswith("ACK sip:503@sip.example.com SIP/2.0\r\n")
    assert "branch=z9hG4bK1" in ack
    assert ";tag=as06d26d3f\r\n" in ack


def test_ack_destination_uses_contact_host_and_port():
    """ACK на 2xx уходит на URI из Contact (RFC 3261 13.2.2.4)."""
    client, _ = make_client()

    assert pyvoip_fixes._ack_destination(
        client, SIPMessage(invite_200_raw())
    ) == ("91.195.230.4", 5060)


def test_ack_destination_falls_back_to_server_without_contact_port():
    client, _ = make_client()
    message = SIPMessage(
        invite_200_raw().replace(
            b"Contact: <sip:+79780481885@91.195.230.4:5060>",
            b"Contact: <sip:+79780481885@sip.example.com>",
        )
    )

    assert pyvoip_fixes._ack_destination(client, message) == (SERVER, 5060)


def test_ack_same_transaction_keeps_invite_branch():
    """``same_transaction`` оставляет branch INVITE — так требует старый Asterisk."""
    client, _ = make_client()
    client.tagLibrary["call-ack-1"] = "localtag"

    ack = client.gen_ack(SIPMessage(invite_200_raw()), same_transaction=True)

    assert ack.startswith("ACK sip:+79780481885@91.195.230.4:5060 SIP/2.0\r\n")
    assert "branch=z9hG4bKold" in ack
    assert "CSeq: 2 ACK\r\n" in ack


def test_repeated_200_ok_gets_extra_ack_with_invite_branch():
    """Повтор 200 OK значит, что первый ACK не принят: шлём ACK транзакции INVITE."""
    client, sock = make_client()
    client.tagLibrary["call-ack-1"] = "localtag"
    pyvoip_fixes._ACK_COUNTS.clear()

    pyvoip_fixes._ack_invite_response(client, SIPMessage(invite_200_raw()))
    assert sock.sent == []  # первый 200 OK подтверждает сам pyVoIP

    pyvoip_fixes._ack_invite_response(client, SIPMessage(invite_200_raw()))

    assert len(sock.sent) == 1
    ack, addr = sock.sent[0]
    assert ack.startswith("ACK sip:+79780481885@91.195.230.4:5060 SIP/2.0\r\n")
    assert "branch=z9hG4bKold" in ack
    assert "To: <sip:+79780481885@sip.example.com>;tag=as53dd332c\r\n" in ack
    assert "From: <sip:503@192.168.1.5>;tag=localtag\r\n" in ack
    assert "CSeq: 2 ACK\r\n" in ack
    assert addr == ("91.195.230.4", 5060)


def test_unknown_404_is_skipped_without_callback(caplog):
    """404 на неизвестный Call-ID (номер не существует) не должен доходить до pyVoIP."""
    client, _ = make_client([register_status_raw(b"404 Not Found", b"1 INVITE")])
    handled = []
    client.callCallback = handled.append

    with caplog.at_level("DEBUG"):
        client.recv()  # не должно бросить исключение
    assert handled == []
    assert "неизвестный Call-ID" in caplog.text


def test_recv_does_not_die_when_handler_raises(caplog):
    """Любая ошибка обработчика не должна убивать поток приёма."""
    client, _ = make_client([register_200_raw()])

    def boom(_message):
        raise KeyError("unknown call-id")

    client.callCallback = boom
    with caplog.at_level("ERROR"):
        client.recv()  # не должно бросить исключение
    assert "Ошибка обработки SIP-сообщения" in caplog.text


# ---------------------------------------------------------------------------
# Регистрация
# ---------------------------------------------------------------------------


def test_register_authenticates():
    client, sock = make_client([register_401_raw(), register_200_raw()])
    assert client._SIPClient__register() is True

    assert len(sock.sent) == 2
    first, second = sock.sent[0][0], sock.sent[1][0]
    assert first.startswith("REGISTER sip:")
    assert "Authorization" not in first
    assert 'Authorization: Digest username="503"' in second
    assert client.s.blocking is True


def test_register_survives_options_between_register_and_answer():
    """OPTIONS во время регистрации не должен сбивать pyVoIP."""
    client, sock = make_client(
        [
            options_raw(),
            register_401_raw(),
            options_raw(call_id="opt-2"),
            register_200_raw(),
        ]
    )
    assert client._SIPClient__register() is True

    replies = [text for text, _ in sock.sent]
    assert replies[0].startswith("REGISTER sip:")
    assert replies[1].startswith("SIP/2.0 200 OK")  # ответ на первый OPTIONS
    assert "Authorization" in replies[2]
    assert replies[3].startswith("SIP/2.0 200 OK")  # ответ на второй OPTIONS


def test_register_raises_on_wrong_credentials():
    client, _ = make_client([register_401_raw(), register_401_raw()])
    with pytest.raises(InvalidAccountInfoError):
        client._SIPClient__register()


def test_register_raises_on_forbidden():
    client, _ = make_client([register_status_raw(b"403 Forbidden", b"1 REGISTER")])
    with pytest.raises(InvalidAccountInfoError):
        client._SIPClient__register()


def test_register_raises_retry_on_500():
    client, _ = make_client([register_status_raw(b"500 Server Internal Error")])
    with pytest.raises(RetryRequiredError):
        client._SIPClient__register()


def test_register_times_out_without_answer():
    client, _ = make_client([])
    with pytest.raises(TimeoutError):
        client._SIPClient__register()


# ---------------------------------------------------------------------------
# install()
# ---------------------------------------------------------------------------


def test_install_is_idempotent():
    pyvoip_fixes.install()
    pyvoip_fixes.install()
    methods = pyVoIP.SIPCompatibleMethods
    for method in pyvoip_fixes.ANSWERABLE_METHODS:
        assert methods.count(method) == 1


def test_install_patches_rtp_transmit_timing():
    """Передатчик RTP должен ждать дедлайн, а не грубый time.sleep."""
    from pyVoIP.RTP import RTPClient, RTPPacketManager

    pyvoip_fixes.install()
    assert RTPClient.trans is pyvoip_fixes._trans
    assert RTPClient.write is pyvoip_fixes._write
    assert RTPClient.encode_packet is pyvoip_fixes._encode_packet
    assert RTPPacketManager.read is pyvoip_fixes._manager_read


# ---------------------------------------------------------------------------
# Исходящий тракт: 16 бит -> G.711 без промежуточного u8
# ---------------------------------------------------------------------------
class FakeRtpSocket:
    """Замена UDP-сокета RTP: складывает отправленные датаграммы."""

    def __init__(self):
        self.sent = []

    def sendto(self, data, addr):
        self.sent.append(bytes(data))
        return len(data)


def make_rtp_client_assoc(assoc):
    """RTPClient с заданным порядком согласованных payload type."""
    from pyVoIP.RTP import RTPClient

    client = RTPClient(
        assoc,
        "127.0.0.1",
        40000,
        "127.0.0.1",
        40002,
        TransmitType.SENDRECV,
    )
    client.sout = FakeRtpSocket()
    return client


def make_rtp_client(payload_type=0):
    """RTPClient с фейковым сокетом (без bind и потоков)."""
    return make_rtp_client_assoc(
        {payload_type: PayloadType(payload_type), 101: PayloadType.EVENT}
    )


def _audioop():
    """audioop без DeprecationWarning (в 3.13 нужен пакет audioop-lts)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import audioop

    return audioop


def snr_db(reference: bytes, measured: bytes) -> float:
    """Отношение сигнал/шум между двумя s16le-потоками, дБ."""
    ref = np.frombuffer(reference, dtype="<i2").astype(np.float64)
    got = np.frombuffer(measured, dtype="<i2").astype(np.float64)
    noise = got - ref
    return 20 * math.log10(np.sqrt(np.mean(ref ** 2)) / np.sqrt(np.mean(noise ** 2)))


def test_rtp_write_keeps_16bit_samples():
    """u8 от приложения превращается в s16 до кодирования G.711."""
    pcm_u8 = audio_utils.tone_s16(440, 20)  # s16-эталон
    u8 = audio_utils.s16_to_u8(pcm_u8)

    client = make_rtp_client()
    client.write(u8)

    stored = client.pmout.read(pyvoip_fixes.PCM16_FRAME_BYTES)
    assert stored == audio_utils.u8_to_s16(u8)


def test_rtp_transmits_pcmu_with_high_snr():
    """24 кГц -> 8 кГц сигнал уходит в μ-law прямо из 16 бит: SNR ~37 дБ."""
    pcm = audio_utils.tone_s16(300, 20, amplitude=0.5)
    u8 = audio_utils.s16_to_u8(pcm)

    client = make_rtp_client(0)
    sequence = client.outSequence
    timestamp = client.outTimestamp
    client.write(u8)
    payload = pyvoip_fixes._transmit_once(client)

    packet = client.sout.sent[-1]
    assert packet[12:] == payload
    assert packet[1] == 0  # PT = PCMU
    assert len(payload) == 160
    assert client.outSequence == (sequence + 1) % 0x10000
    assert client.outTimestamp == (timestamp + 160) % 0x100000000

    decoded = _audioop().ulaw2lin(payload, 2)
    assert snr_db(audio_utils.u8_to_s16(u8), decoded) >= 34.0


def test_rtp_transmits_pcma_as_alaw():
    """PCMA кодируется A-law, а не μ-law (баг pyVoIP 1.6.8)."""
    pcm = audio_utils.tone_s16(300, 20, amplitude=0.5)
    u8 = audio_utils.s16_to_u8(pcm)
    pcm16 = audio_utils.u8_to_s16(u8)

    client = make_rtp_client(8)
    client.write(u8)
    payload = pyvoip_fixes._transmit_once(client)

    assert client.sout.sent[-1][1] == 8  # PT = PCMA
    assert payload == audio_utils.s16_to_alaw(pcm16)
    assert payload != audio_utils.s16_to_ulaw(pcm16)

    decoded = _audioop().alaw2lin(payload, 2)
    assert snr_db(pcm16, decoded) >= 34.0


def test_rtp_underrun_is_16bit_silence():
    """Пустой исходящий буфер добивается нулями, а не 0x80 (это -32640)."""
    client = make_rtp_client()
    payload = pyvoip_fixes._transmit_once(client)

    assert payload == b"\xff" * 160  # μ-law нуля
    assert _audioop().ulaw2lin(payload, 2) == b"\x00\x00" * 160


def test_manager_read_pads_silence_per_format():
    """Буфер входящего аудио по-прежнему добивается 0x80."""
    from pyVoIP.RTP import RTPPacketManager

    assert RTPPacketManager().read(10) == b"\x80" * 10


def test_encode_packet_uses_negotiated_codec():
    """encode_packet маршрутизирует по PT, а не всегда в μ-law."""
    pcm16 = audio_utils.tone_s16(500, 20)
    client = make_rtp_client(0)
    assert client.encode_packet(pcm16) == audio_utils.s16_to_ulaw(pcm16)

    alaw_client = make_rtp_client(8)
    assert alaw_client.encode_packet(pcm16) == audio_utils.s16_to_alaw(pcm16)


def test_select_audio_codec_prefers_g711():
    """Из согласованных кодеков выбираем G.711, а не первый payload из SDP."""
    client = make_rtp_client_assoc(
        {18: PayloadType.G729, 0: PayloadType.PCMU, 8: PayloadType.PCMA}
    )
    assert client.preference is PayloadType.G729
    assert pyvoip_fixes.select_audio_codec(client) is PayloadType.PCMU

    alaw_only = make_rtp_client_assoc({8: PayloadType.PCMA})
    assert pyvoip_fixes.select_audio_codec(alaw_only) is PayloadType.PCMA

    unsupported = make_rtp_client_assoc({18: PayloadType.G729})
    assert pyvoip_fixes.select_audio_codec(unsupported) is None


def test_rtp_trans_uses_pcmu_when_sdp_offers_g729_first(monkeypatch):
    """Передатчик не умирает, если первым в SDP идёт кодек без G.711."""
    client = make_rtp_client_assoc(
        {18: PayloadType.G729, 0: PayloadType.PCMU}
    )
    client.write(audio_utils.s16_to_u8(audio_utils.tone_s16(300, 20)))

    def stop_after_packet(_deadline):
        client.NSD = False

    monkeypatch.setattr(pyvoip_fixes, "wait_until", stop_after_packet)
    pyvoip_fixes._trans(client)

    assert client.preference is PayloadType.PCMU
    assert client.sout.sent, "передатчик должен отправить хотя бы один пакет"
    assert client.sout.sent[0][1] == 0  # PT = PCMU


def test_rtp_trans_stops_without_g711(monkeypatch, caplog):
    """Если G.711 не согласован, пишем понятную ошибку и не шлём мусор."""
    client = make_rtp_client_assoc({101: PayloadType.EVENT})

    monkeypatch.setattr(
        pyvoip_fixes,
        "wait_until",
        lambda _deadline: pytest.fail("передатчик не должен начинать цикл"),
    )
    with caplog.at_level(logging.ERROR, logger="sipbot.pyvoip_fixes"):
        pyvoip_fixes._trans(client)

    assert not client.sout.sent
    assert "PCMU/PCMA" in caplog.text


def response_raw(
    status_line: bytes,
    call_id: str = "call-1@192.168.1.5:5060",
    extra: bytes = b"",
) -> bytes:
    return (
        b"SIP/2.0 " + status_line + b"\r\n"
        b"Via: SIP/2.0/UDP 192.168.1.5:5060;branch=z9hG4bK1;rport=5060\r\n"
        b'From: "503" <sip:503@sip.example.com>;tag=a8c9c62d\r\n'
        b'To: "503" <sip:sip.example.com>;tag=as06d26d3f\r\n'
        b"Call-ID: " + call_id.encode() + b"\r\n"
        b"CSeq: 1 INVITE\r\n"
        + extra
        + b"Content-Length: 0\r\n\r\n"
    )


INVITE_MEDIA = {40000: {0: PayloadType.PCMU}}


def _header_value(raw: str, name: str) -> str:
    """Значение SIP-заголовка (или параметра ``branch=`` из ``Via``)."""
    if name == "branch":
        assert "branch=" in raw
        return raw.split("branch=", 1)[1].split(";", 1)[0].split("\r\n", 1)[0]
    for line in raw.split("\r\n"):
        if line.lower().startswith(name.lower() + ":"):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"в запросе нет заголовка {name}")


def test_invite_returns_on_session_progress(monkeypatch):
    """183 Session Progress не должен держать call() внутри recv()."""
    call_id = "test-call-id"
    client, sock = make_client(
        [response_raw(b"183 Session Progress", call_id)]
    )
    monkeypatch.setattr(client, "gen_call_id", lambda: call_id)

    _, got_call_id, _ = pyvoip_fixes._invite(
        client, "100", INVITE_MEDIA, TransmitType.SENDRECV
    )

    assert got_call_id == call_id
    assert sock.sent[0][0].startswith("INVITE sip:100@")


def test_invite_returns_without_any_response(monkeypatch):
    """Если АТС молчит, INVITE не блокирует вызывающий поток навсегда."""
    monkeypatch.setattr(pyvoip_fixes, "INVITE_FIRST_RESPONSE_TIMEOUT", 0.05)
    client, sock = make_client()

    _, call_id, _ = pyvoip_fixes._invite(
        client, "100", INVITE_MEDIA, TransmitType.SENDRECV
    )

    assert call_id
    assert sock.sent and sock.sent[0][0].startswith("INVITE ")


def test_invite_authorizes_on_401(monkeypatch):
    """401 на INVITE: ACK, затем повтор **новой транзакцией** с Digest.

    Если переслать тот же INVITE с добавленным заголовком (как делал патч
    раньше), branch и CSeq остаются прежними: Asterisk считает пакет
    ретрансмиссией, повторяет 401 и звонок висит до таймаута.
    """
    call_id = "test-call-id"
    challenge = response_raw(
        b"401 Unauthorized",
        call_id,
        extra=(
            b'WWW-Authenticate: Digest algorithm=MD5, realm="asterisk", '
            b'nonce="abc123"\r\n'
        ),
    )
    client, sock = make_client([challenge])
    monkeypatch.setattr(client, "gen_call_id", lambda: call_id)

    request, _, _ = pyvoip_fixes._invite(
        client, "100", INVITE_MEDIA, TransmitType.SENDRECV
    )

    texts = [data for data, _ in sock.sent]
    assert texts[0].startswith("INVITE ")
    assert texts[1].startswith("ACK ")  # не-2xx финальный ответ обязан на ACK
    retry = next(text for text in texts if "Authorization: Digest" in text)

    assert _header_value(retry, "CSeq") != _header_value(texts[0], "CSeq")
    assert _header_value(retry, "branch") != _header_value(texts[0], "branch")
    assert _header_value(retry, "Call-ID") == call_id
    # вызывающий код получает именно авторизованный INVITE: иначе ACK на
    # 200 OK уйдёт с прежним CSeq
    assert request.raw.startswith(retry.encode("utf8")[:60])


def test_invite_authorizes_on_407(monkeypatch):
    """407 Proxy Authentication Required: заголовок Proxy-Authorization."""
    call_id = "test-call-id"
    challenge = response_raw(
        b"407 Proxy Authentication Required",
        call_id,
        extra=(
            b'Proxy-Authenticate: Digest realm="proxy", nonce="xyz789"\r\n'
        ),
    )
    client, sock = make_client([challenge])
    monkeypatch.setattr(client, "gen_call_id", lambda: call_id)

    pyvoip_fixes._invite(client, "100", INVITE_MEDIA, TransmitType.SENDRECV)

    texts = [data for data, _ in sock.sent]
    retry = next(
        text for text in texts if "Proxy-Authorization: Digest" in text
    )
    assert 'nonce="xyz789"' in retry
    assert _header_value(retry, "branch") != _header_value(texts[0], "branch")


def test_unhandled_response_is_logged(caplog):
    """Отказ АТС (403/407/...) виден в логе, а не только в stdout pyVoIP."""
    client, _ = make_client()
    message = SIPMessage(response_raw(b"403 Forbidden"))

    with caplog.at_level(logging.WARNING, logger="sipbot.pyvoip_fixes"):
        pyvoip_fixes._parse_message(client, message)

    assert "403" in caplog.text


def test_session_progress_is_not_a_warning(caplog):
    """183 Session Progress — нормальный ответ, шуметь о нём не нужно."""
    client, _ = make_client()
    message = SIPMessage(response_raw(b"183 Session Progress"))

    with caplog.at_level(logging.DEBUG, logger="sipbot.pyvoip_fixes"):
        pyvoip_fixes._parse_message(client, message)

    assert "183" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_codec_is_logged(caplog):
    """В лог пишется согласованный кодек — это диагностика проблем со звуком."""
    client = make_rtp_client(0)
    with caplog.at_level(logging.INFO, logger="sipbot.pyvoip_fixes"):
        pyvoip_fixes._log_codec(client)
    assert "PCMU" in caplog.text


def test_tx_tap_writes_what_goes_to_network(tmp_path, monkeypatch):
    """SIPBOT_TX_WAV сохраняет ровно отправленные 16-битные кадры."""
    target = tmp_path / "tx.wav"
    monkeypatch.setenv("SIPBOT_TX_WAV", str(target))
    pcm = audio_utils.tone_s16(440, 20)
    u8 = audio_utils.s16_to_u8(pcm)

    client = make_rtp_client()
    client.write(u8)
    pyvoip_fixes._transmit_once(client)
    pyvoip_fixes._close_tx_tap(client)

    with wave.open(str(target), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 8000
        assert wav.readframes(wav.getnframes()) == audio_utils.u8_to_s16(u8)

