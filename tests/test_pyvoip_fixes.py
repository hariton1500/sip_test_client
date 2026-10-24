"""Тесты патчей pyVoIP: ответ на OPTIONS и устойчивая регистрация.

Всё оффлайн: вместо UDP-сокета — фейк с заранее заготовленными пакетами,
вместо select — заглушка, которая «готова», только если в очереди есть данные.
Сырые сообщения взяты из реального обмена с Asterisk 11.7.
"""

from types import SimpleNamespace

import pyVoIP
import pytest
from pyVoIP.SIP import (
    InvalidAccountInfoError,
    RetryRequiredError,
    SIPClient,
    SIPMessage,
    SIPStatus,
)

from sipbot import pyvoip_fixes

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
