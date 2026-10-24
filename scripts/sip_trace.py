"""Трассировка SIP/RTP pyVoIP на уровне сокетов (диагностика live-тестов).

Печатает каждое отправленное/полученное SIP-сообщение с адресом и таймстампом,
RTP-пакеты по умолчанию только считает (``--rtp`` — печатать).

Запуск:
    python scripts/sip_trace.py 15                 # 15 секунд регистрации
    python scripts/sip_trace.py 15 --rtp           # плюс каждый RTP-пакет
    python scripts/sip_trace.py 15 --debug --stack # хуки register() + стек
"""

from __future__ import annotations

import socket
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sipbot.config import Settings  # noqa: E402
from sipbot.phone import SipPhone  # noqa: E402

_real_socket = socket.socket

show_rtp = False
show_stack = False
rtp_in = 0
rtp_out = 0
sip_in = 0
sip_out = 0


def _stamp() -> str:
    return time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"


def _looks_like_sip(payload: bytes) -> bool:
    if payload[:4] == b"SIP/":
        return True
    head = payload[:32]
    return head[:3].isalpha() and (b" sip:" in head or head.endswith(b" sip:"))


def _dump(direction: str, data: bytes, addr: Optional[Tuple[str, int]]) -> None:
    global rtp_in, rtp_out, sip_in, sip_out
    if _looks_like_sip(data):
        if direction == ">>>":
            sip_out += 1
        else:
            sip_in += 1
        text = data.decode("utf-8", "replace")
        thread = threading.current_thread().name
        print(f"\n{_stamp()} {direction} {addr} [{thread}]\n{text}", flush=True)
        if show_stack and direction == ">>>":
            stack = traceback.format_stack(limit=8)
            print("   stack:\n" + "".join(stack), flush=True)
    else:
        if direction == ">>>":
            rtp_out += 1
        else:
            rtp_in += 1
        if show_rtp:
            print(
                f"{_stamp()} {direction} RTP {addr} len={len(data)} "
                f"head={data[:12].hex()}",
                flush=True,
            )


class TracingSocket(_real_socket):
    """Сокет с логированием sendto/recvfrom (SIP и RTP идут через него)."""

    def sendto(self, data, *args):  # type: ignore[override]
        _dump(">>>", bytes(data), args[-1] if args else None)
        return super().sendto(data, *args)

    def recvfrom(self, *args, **kwargs):  # type: ignore[override]
        data, addr = super().recvfrom(*args, **kwargs)
        _dump("<<<", bytes(data), addr)
        return data, addr

    def recv(self, *args, **kwargs):  # type: ignore[override]
        data = super().recv(*args, **kwargs)
        _dump("<<<", bytes(data), None)
        return data


def _install_pyvoip_hooks() -> None:
    """Логирует вход в register() и создание таймера перерегистрации."""
    import pyVoIP.SIP as pyvoip_sip

    orig_register = pyvoip_sip.SIPClient.register
    orig_timer = pyvoip_sip.SIPClient._SIPClient__start_register_timer
    orig_parse = pyvoip_sip.SIPClient.parse_message

    def register(self):
        print(
            f"# register() enter thread={threading.current_thread().name} "
            f"failures={self.registerFailures}",
            flush=True,
        )
        try:
            result = orig_register(self)
        except BaseException as exc:  # noqa: BLE001
            print(f"# register() raise {exc!r}", flush=True)
            raise
        print(f"# register() exit -> {result}", flush=True)
        return result

    def start_timer(self, delay=None):
        counter = self.registerCounter.x
        print(
            f"# start_register_timer(delay={delay}) counter={counter} "
            f"thread={threading.current_thread().name}",
            flush=True,
        )
        return orig_timer(self, delay)

    def parse_message(self, message):
        try:
            kind = f"{message.method} {message.status}"
        except Exception:  # noqa: BLE001
            kind = "?"
        print(
            f"# parse_message({kind}) thread={threading.current_thread().name}",
            flush=True,
        )
        return orig_parse(self, message)

    pyvoip_sip.SIPClient.register = register
    pyvoip_sip.SIPClient._SIPClient__start_register_timer = start_timer
    pyvoip_sip.SIPClient.parse_message = parse_message


def main(argv: Optional[list] = None) -> int:
    global show_rtp, show_stack
    args = list(sys.argv[1:] if argv is None else argv)
    show_rtp = "--rtp" in args
    show_stack = "--stack" in args
    debug_hooks = "--debug" in args
    args = [a for a in args if a not in ("--rtp", "--stack", "--debug")]
    if debug_hooks:
        _install_pyvoip_hooks()
    hold = float(args[0]) if args else 10.0

    settings = Settings.from_env(".env")
    settings.validate()
    print(f"# {settings.describe()}", flush=True)

    socket.socket = TracingSocket  # type: ignore[misc]
    started = time.monotonic()
    phone = SipPhone(settings)
    phone.start()
    status = phone.wait_registered(min(hold, 20.0))
    print(f"# статус: {status.name}", flush=True)
    remaining = hold - (time.monotonic() - started)
    if remaining > 0:
        time.sleep(remaining)
    print(f"# итог: {phone.status().name}", flush=True)
    phone.stop()
    socket.socket = _real_socket  # type: ignore[misc]
    print(
        f"# пакетов: SIP отправлено {sip_out}, получено {sip_in}; "
        f"RTP отправлено {rtp_out}, получено {rtp_in}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
