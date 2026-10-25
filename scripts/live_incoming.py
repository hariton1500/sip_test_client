#!/usr/bin/env python3
"""Живой тест одного входящего звонка на настоящем клиенте.

Скрипт поднимает на локальном IPv4-интерфейсе минимальную SIP-АТС и
вызывающего абонента (реальные UDP-сокеты, настоящий SIP/RTP), запускает
``python main.py -v incoming`` отдельным процессом, звонит на него ровно один
раз, обменивается аудио в обе стороны, кладёт трубку и останавливает клиент.

Зачем нужен отдельный harness: у машины нет второй SIP-линии, а входящий
звонок с внешней АТС требует проброса портов. Здесь обмен идёт по локальной
сети (SIP_MY_IP — адрес интерфейса, не loopback), поэтому проверяются ровно
те же ветки кода pyVoIP/sipbot, что и с настоящей АТС: REGISTER, INVITE,
180/200, SDP, RTP в обе стороны, BYE, завершение сессии и запись WAV.

Пример:

    .venv/bin/python scripts/live_incoming.py

Результаты: ``recordings/live_incoming_one/`` (client.log, caller_heard.wav,
report.txt).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sipbot.audio_utils import (  # noqa: E402
    FRAME_BYTES,
    SILENCE_BYTE,
    SILENCE_FRAME,
    iter_frames,
    read_wav,
    resample_s16,
    rms_u8,
    s16_to_u8,
    tone_u8,
    u8_to_s16,
    ulaw_decode,
    ulaw_encode,
    write_wav,
)

logger = logging.getLogger("live_incoming")
RTP_HEADER = 12


# ---------------------------------------------------------------------------
# Минимальный разбор/сборка SIP
# ---------------------------------------------------------------------------
@dataclass
class SipMessage:
    """Разобранное SIP-сообщение (только то, что нужно harness'у)."""

    start: str
    headers: Dict[str, List[str]]
    body: str

    @property
    def is_response(self) -> bool:
        return self.start.startswith("SIP/2.0")

    @property
    def status(self) -> int:
        return int(self.start.split(" ", 2)[1]) if self.is_response else 0

    @property
    def method(self) -> str:
        return self.start.split(" ", 2)[0].upper()

    @property
    def cseq(self) -> str:
        return self.get("CSeq")

    @property
    def cseq_method(self) -> str:
        parts = self.cseq.split()
        return parts[1].upper() if len(parts) > 1 else ""

    def get(self, name: str, default: str = "") -> str:
        values = self.headers.get(name.lower())
        return values[0] if values else default

    def all(self, name: str) -> List[str]:
        return list(self.headers.get(name.lower(), []))


def parse_sip(raw: bytes) -> SipMessage:
    text = raw.decode("utf-8", "replace")
    head, _, body = text.partition("\r\n\r\n")
    lines = head.split("\r\n")
    headers: Dict[str, List[str]] = {}
    for line in lines[1:]:
        if not line or line[:1] in " \t":
            continue
        name, sep, value = line.partition(":")
        if not sep:
            continue
        headers.setdefault(name.strip().lower(), []).append(value.strip())
    return SipMessage(start=lines[0].strip(), headers=headers, body=body)


def _append_tag(value: str, tag: str) -> str:
    if not tag or ";tag=" in value.lower():
        return value
    return f"{value};tag={tag}"


def build_response(
    request: SipMessage, status: int, phrase: str, to_tag: str = ""
) -> str:
    """Собирает ответ на запрос, сохраняя Via/From/CSeq вызывающего."""
    lines = [f"SIP/2.0 {status} {phrase}"]
    lines.extend(f"Via: {via}" for via in request.all("Via"))
    lines.append(f"From: {request.get('From')}")
    lines.append(f"To: {_append_tag(request.get('To'), to_tag)}")
    lines.append(f"Call-ID: {request.get('Call-ID')}")
    lines.append(f"CSeq: {request.get('CSeq')}")
    contact = request.get("Contact")
    if contact:
        lines.append(f"Contact: {contact}")
    lines.extend(["User-Agent: live-incoming-harness", "Content-Length: 0"])
    return "\r\n".join(lines) + "\r\n\r\n"


def parse_sdp(body: str) -> Tuple[Optional[str], Optional[int]]:
    """Возвращает (IP из c=, порт из m=audio) ответа клиента."""
    address: Optional[str] = None
    port: Optional[int] = None
    for line in body.replace("\r\n", "\n").splitlines():
        if line.startswith("c=IN IP4 "):
            address = line.split()[2]
        elif line.startswith("m=audio "):
            port = int(line.split()[1].split("/")[0])
    return address, port


def contact_uri(message: SipMessage, fallback: str) -> str:
    contact = message.get("Contact")
    if "<" in contact and ">" in contact:
        return contact.split("<", 1)[1].split(">", 1)[0]
    return contact or fallback



# ---------------------------------------------------------------------------
# SIP-УАС: регистрация, один INVITE, ответы на BYE/REGISTER
# ---------------------------------------------------------------------------
class SipUas(threading.Thread):
    """Минимальная «АТС + вызывающий» на одном UDP-сокете."""

    def __init__(
        self,
        sock: socket.socket,
        my_ip: str,
        port: int,
        client_user: str,
        client_sip_port: int,
        caller_user: str,
    ) -> None:
        super().__init__(name="live-sip-uas", daemon=True)
        self.sock = sock
        self.my_ip = my_ip
        self.port = port
        self.client_user = client_user
        self.client_sip_port = client_sip_port
        self.caller_user = caller_user

        self.registered = threading.Event()
        self.ringing = threading.Event()
        self.answered = threading.Event()
        self.bye_ok = threading.Event()
        self.deregistered = threading.Event()

        self.answer: Optional[SipMessage] = None
        self.answer_addr: Optional[Tuple[str, int]] = None
        self.client_ip: Optional[str] = None
        self.client_rtp_port: Optional[int] = None

        self.invites = 0
        self.invites_sent = 0
        self.registers = 0
        self.trace: List[str] = []

        self._stop = threading.Event()
        self._reg_tag = f"srv{uuid.uuid4().hex[:6]}"
        self._lock = threading.Lock()

    # -- журнал ---------------------------------------------------------
    def _log(self, direction: str, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with self._lock:
            self.trace.append(f"{stamp} {direction} {text}")

    # -- отправка -------------------------------------------------------
    def _send(self, text: str, addr: Tuple[str, int]) -> None:
        self.sock.sendto(text.encode("utf-8"), addr)

    def send_invite(self, call_id: str, from_tag: str, rtp_port: int) -> None:
        """Отправляет единственный INVITE клиенту."""
        to_uri = f"sip:{self.client_user}@{self.my_ip}"
        from_uri = f"sip:{self.caller_user}@{self.my_ip}"
        session = str(random.randint(1, 999999))
        body = (
            "v=0\r\n"
            f"o=- {session} {session} IN IP4 {self.my_ip}\r\n"
            "s=live-incoming-harness\r\n"
            f"c=IN IP4 {self.my_ip}\r\n"
            "t=0 0\r\n"
            f"m=audio {rtp_port} RTP/AVP 0\r\n"
            "a=rtpmap:0 PCMU/8000\r\n"
            "a=ptime:20\r\n"
            "a=maxptime:150\r\n"
            "a=sendrecv\r\n"
        )
        branch = f"z9hG4bK{uuid.uuid4().hex[:24]}"
        invite = (
            f"INVITE {to_uri} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {self.my_ip}:{self.port};branch={branch};rport\r\n"
            "Max-Forwards: 70\r\n"
            f"From: <{from_uri}>;tag={from_tag}\r\n"
            f"To: <{to_uri}>\r\n"
            f"Call-ID: {call_id}\r\n"
            "CSeq: 1 INVITE\r\n"
            f"Contact: <{from_uri}>\r\n"
            "Allow: INVITE, ACK, CANCEL, BYE, OPTIONS\r\n"
            "User-Agent: live-incoming-harness\r\n"
            "Content-Type: application/sdp\r\n"
            f"Content-Length: {len(body.encode('utf-8'))}\r\n"
            "\r\n"
            f"{body}"
        )
        self._log("-->", f"INVITE {call_id} (RTP {rtp_port})")
        self.invites_sent += 1
        self._send(invite, (self.my_ip, self.client_sip_port))

    def send_bye(self, call_id: str) -> None:
        assert self.answer is not None and self.answer_addr is not None
        target_uri = contact_uri(
            self.answer,
            f"sip:{self.client_user}@{self.client_ip}:{self.client_sip_port}",
        )
        branch = f"z9hG4bK{uuid.uuid4().hex[:24]}"
        bye = (
            f"BYE {target_uri} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {self.my_ip}:{self.port};branch={branch};rport\r\n"
            "Max-Forwards: 70\r\n"
            f"From: {self.answer.get('From')}\r\n"
            f"To: {self.answer.get('To')}\r\n"
            f"Call-ID: {call_id}\r\n"
            "CSeq: 2 BYE\r\n"
            "User-Agent: live-incoming-harness\r\n"
            "Content-Length: 0\r\n"
            "\r\n"
        )
        self._log("-->", f"BYE {call_id}")
        self._send(bye, self.answer_addr)

    def _send_ack(self, response: SipMessage, addr: Tuple[str, int]) -> None:
        target = contact_uri(
            response,
            f"sip:{self.client_user}@{self.client_ip}:{self.client_sip_port}",
        )
        branch = f"z9hG4bK{uuid.uuid4().hex[:24]}"
        ack = (
            f"ACK {target} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {self.my_ip}:{self.port};branch={branch};rport\r\n"
            "Max-Forwards: 70\r\n"
            f"From: {response.get('From')}\r\n"
            f"To: {response.get('To')}\r\n"
            f"Call-ID: {response.get('Call-ID')}\r\n"
            f"CSeq: {response.cseq.split()[0]} ACK\r\n"
            "User-Agent: live-incoming-harness\r\n"
            "Content-Length: 0\r\n"
            "\r\n"
        )
        self._log("-->", "ACK")
        self._send(ack, addr)

    # -- приём ----------------------------------------------------------
    def run(self) -> None:
        while not self._stop.is_set():
            try:
                raw, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            if not raw:
                continue
            try:
                message = parse_sip(raw)
            except Exception:  # noqa: BLE001
                self._log("<--", f"неразобранный пакет {len(raw)} байт")
                continue
            try:
                self._handle(message, addr)
            except Exception:  # noqa: BLE001
                logger.exception("Ошибка обработки SIP: %s", message.start)
                self._log("<--", f"ошибка обработки: {message.start}")

    def _handle(self, message: SipMessage, addr: Tuple[str, int]) -> None:
        self._log("<--", f"{message.start} (CSeq {message.cseq or '-'})")

        if message.is_response:
            status = message.status
            method = message.cseq_method
            if method == "INVITE" and status == 180:
                self.ringing.set()
            elif method == "INVITE" and status == 200:
                self.answer = message
                self.answer_addr = addr
                self.client_ip, self.client_rtp_port = parse_sdp(message.body)
                self._send_ack(message, addr)
                self.answered.set()
            elif method == "BYE" and status == 200:
                self.bye_ok.set()
            return

        method = message.method
        if method == "REGISTER":
            self.registers += 1
            self._send(build_response(message, 200, "OK", self._reg_tag), addr)
            if message.get("Expires").strip() == "0":
                self.deregistered.set()
            else:
                self.registered.set()
        elif method == "INVITE":
            self.invites += 1
            if not self.ringing.is_set():
                self._send(
                    build_response(message, 180, "Ringing", self._reg_tag), addr
                )
                self.ringing.set()
        elif method == "ACK":
            pass
        elif method == "BYE":
            self._send(build_response(message, 200, "OK"), addr)
        elif method in ("OPTIONS", "INFO", "NOTIFY", "UPDATE", "CANCEL"):
            self._send(build_response(message, 200, "OK"), addr)

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# RTP и вспомогательные функции
# ---------------------------------------------------------------------------
@dataclass
class RtpStats:
    sent: int = 0
    received: int = 0
    payload_bytes: int = 0
    out_of_order: int = 0
    first_seq: Optional[int] = None
    last_seq: Optional[int] = None
    pcm_u8: bytearray = field(default_factory=bytearray)

    @property
    def duration_ms(self) -> float:
        return self.payload_bytes * 1000.0 / 8000.0


def detect_ip() -> str:
    """Определяет IPv4 интерфейса, через который машина выходит в сеть."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        ip = probe.getsockname()[0]
    except OSError:
        ip = ""
    finally:
        probe.close()
    if ip and not ip.startswith("127."):
        return ip
    try:
        for info in socket.getaddrinfo(
            socket.gethostname(), None, socket.AF_INET
        ):
            candidate = info[4][0]
            if not candidate.startswith("127."):
                return candidate
    except OSError:
        pass
    return ""


def bind_udp(ip: str, port: int = 0) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((ip, port))
    sock.setblocking(False)
    return sock


def run_rtp(
    sock: socket.socket,
    destination: Tuple[str, int],
    duration: float,
    tone_payloads: List[bytes],
) -> RtpStats:
    """Говорит тоном первые ~1.8 с, затем тишиной; пишет всё от клиента."""
    stats = RtpStats()
    silence = ulaw_encode(SILENCE_FRAME)
    sequence = random.randint(1, 60000)
    timestamp = random.randint(1, 100000)
    ssrc = random.randint(1000, 65000)

    start = time.monotonic()
    next_send = start
    sent_media = 0.0
    while True:
        now = time.monotonic()
        while next_send <= now and sent_media < duration:
            payload = (
                tone_payloads[stats.sent]
                if stats.sent < len(tone_payloads)
                else silence
            )
            packet = (
                struct.pack(
                    "!BBHII",
                    0x2 << 6,
                    0,
                    sequence & 0xFFFF,
                    timestamp & 0xFFFFFFFF,
                    ssrc,
                )
                + payload
            )
            try:
                sock.sendto(packet, destination)
            except OSError as exc:
                logger.warning("RTP sendto: %s", exc)
            stats.sent += 1
            sequence += 1
            timestamp += len(payload)
            next_send += FRAME_BYTES / 8000.0
            sent_media += FRAME_BYTES / 8000.0

        now = time.monotonic()
        remaining = duration - (now - start)
        if remaining <= 0:
            break
        timeout = max(0.0, min(0.02, next_send - now, remaining))
        readable, _, _ = select.select([sock], [], [], timeout)
        if not readable:
            continue
        while True:
            try:
                data, _ = sock.recvfrom(4096)
            except BlockingIOError:
                break
            except OSError:
                return stats
            if len(data) <= RTP_HEADER:
                continue
            if (data[1] & 0x7F) != 0:  # ждём только PCMU (payload type 0)
                continue
            seq = int.from_bytes(data[2:4], "big")
            if stats.last_seq is not None and seq != (stats.last_seq + 1) & 0xFFFF:
                stats.out_of_order += 1
            if stats.first_seq is None:
                stats.first_seq = seq
            stats.last_seq = seq
            payload = data[RTP_HEADER:]
            stats.payload_bytes += len(payload)
            stats.received += 1
            stats.pcm_u8.extend(ulaw_decode(payload))
    return stats


def analyze_pcm(pcm_u8: bytes) -> Dict[str, float]:
    """Уровень и число «звучных» 20-мс кадров в записанном аудио."""
    frames = 0
    voiced = 0
    peak = 0.0
    for frame in iter_frames(bytes(pcm_u8), FRAME_BYTES, SILENCE_BYTE):
        frames += 1
        level = rms_u8(frame)
        peak = max(peak, level)
        if level > 0.02:
            voiced += 1
    return {
        "frames": float(frames),
        "voiced": float(voiced),
        "peak": peak,
        "duration_ms": frames * FRAME_BYTES * 1000.0 / 8000.0,
    }


def read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def wait_for_log(
    path: Path, needle: str, timeout: float, proc: Optional[subprocess.Popen] = None
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if needle in read_log(path):
            return True
        if proc is not None and proc.poll() is not None:
            return needle in read_log(path)
        time.sleep(0.1)
    return needle in read_log(path)


def log_key_lines(text: str) -> List[str]:
    keys = (
        "Статус регистрации",
        "Многоканальный",
        "Вызов принят",
        "Звонок завершён",
        "Запись сохранена",
        "Traceback",
        "ERROR",
        "WARNING",
        "RTP Packet failed",
    )
    lines = []
    for line in text.splitlines():
        if any(key in line for key in keys):
            lines.append(line)
    return lines


def fetch_ui_messages(port: int, timeout: float = 3.0) -> List[dict]:
    """Читает историю сообщений из локального веб-интерфейса."""
    url = f"http://127.0.0.1:{port}/api/messages"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        messages = payload.get("messages") or []
        return [m for m in messages if isinstance(m, dict)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не удалось прочитать UI %s: %s", url, exc)
        return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Живой тест одного входящего звонка (локальный SIP+RTP)"
    )
    parser.add_argument(
        "--my-ip",
        default="",
        help="IPv4 интерфейса для SIP/RTP (по умолчанию определяется сам)",
    )
    parser.add_argument("--sip-port", type=int, default=15060, help="порт harness-АТС")
    parser.add_argument(
        "--client-sip-port", type=int, default=15061, help="локальный SIP-порт клиента"
    )
    parser.add_argument(
        "--client-user",
        default=os.environ.get("SIP_USER", "503"),
        help="SIP-логин клиента (по умолчанию из SIP_USER или 503)",
    )
    parser.add_argument(
        "--caller-user", default="508", help="номер вызывающего абонента"
    )
    parser.add_argument(
        "--rtp-low",
        type=int,
        default=12100,
        help="нижняя граница RTP-портов клиента",
    )
    parser.add_argument(
        "--rtp-high",
        type=int,
        default=12120,
        help="верхняя граница RTP-портов клиента",
    )
    parser.add_argument(
        "--bot",
        default="tone",
        help=(
            "бот клиента: tone|placeholder|echo|silence|ai "
            "(по умолчанию tone; ai требует OPENAI_* и TTS)"
        ),
    )
    parser.add_argument(
        "--caller-wav",
        default="",
        help="WAV, который вызывающий отправляет клиенту (для проверки ASR)",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="включить веб-интерфейс клиента с живой расшифровкой",
    )
    parser.add_argument(
        "--ui-port", type=int, default=8765, help="порт веб-интерфейса клиента"
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        help="не включать ASR в клиенте (UI покажет только события)",
    )
    parser.add_argument(
        "--ui-hold",
        type=float,
        default=0.0,
        help="держать клиент и UI после звонка, с (для просмотра в браузере)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="длительность разговора после ответа, с",
    )
    parser.add_argument(
        "--record-dir",
        default=str(ROOT / "recordings" / "live_incoming_one"),
        help="каталог результатов",
    )
    parser.add_argument(
        "--keep", action="store_true", help="не очищать каталог результатов"
    )
    parser.add_argument(
        "--reg-timeout", type=float, default=25.0, help="ожидание регистрации, с"
    )
    return parser


def wav_info(path: Path) -> str:
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            seconds = frames / rate if rate else 0.0
            return (
                f"{path.name}: {path.stat().st_size} байт, "
                f"{frames} кадров, {rate} Гц, {seconds:.1f} с"
            )
    except Exception as exc:  # noqa: BLE001
        return f"{path.name}: не читается ({exc})"


def write_report(
    record_dir: Path,
    my_ip: str,
    args: argparse.Namespace,
    uas: SipUas,
    rtp: Optional[RtpStats],
    analysis: Optional[Dict[str, float]],
    recording: Optional[Path],
    caller_wav: Path,
    checks: List[Tuple[bool, str]],
    log_text: str,
) -> Tuple[Path, bool]:
    passed = all(ok for ok, _ in checks) and bool(checks)

    lines = [
        "# Живой тест одного входящего звонка",
        f"Время: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Конфигурация",
        f"- harness (АТС+вызывающий): {my_ip}:{args.sip_port}",
        f"- клиент: {args.client_user} @ {my_ip}:{args.client_sip_port}",
        f"- RTP-порты клиента: {args.rtp_low}-{args.rtp_high}",
        f"- вызывающий: {args.caller_user}",
        f"- бот клиента: {args.bot}, разговор: {args.duration:.1f} с",
        "",
        "## SIP-обмен",
        *[f"    {line}" for line in uas.trace],
        "",
        "## RTP",
    ]
    if rtp is None:
        lines.append("    обмена не было")
    else:
        lines.extend(
            [
                f"    отправлено вызывающим: {rtp.sent} пакетов",
                f"    получено от клиента: {rtp.received} пакетов, "
                f"{rtp.payload_bytes} байт ({rtp.duration_ms / 1000.0:.1f} с)",
                f"    sequence: {rtp.first_seq}..{rtp.last_seq}, "
                f"нарушений порядка: {rtp.out_of_order}",
            ]
        )
    lines.append("")
    lines.append("## Аудио, услышанное вызывающим (client -> caller)")
    if analysis is None:
        lines.append("    записи нет")
    else:
        lines.extend(
            [
                f"    кадров: {int(analysis['frames'])}, "
                f"длительность: {analysis['duration_ms'] / 1000.0:.1f} с",
                f"    «звучных» кадров: {int(analysis['voiced'])}, "
                f"пиковый RMS: {analysis['peak']:.3f}",
                f"    файл: {wav_info(caller_wav)}",
            ]
        )
    lines.append("")
    lines.append("## Запись клиента (caller -> client)")
    lines.append(f"    {wav_info(recording) if recording else 'файл не создан'}")
    lines.append("")
    lines.append("## Проверки")
    for ok, text in checks:
        lines.append(f"    [{'OK' if ok else 'FAIL'}] {text}")
    lines.append("")
    lines.append("## Ключевые строки client.log")
    key_lines = log_key_lines(log_text)
    if key_lines:
        lines.extend(f"    {line}" for line in key_lines)
    else:
        lines.append("    (нет)")
    lines.extend(["", f"ИТОГ: {'PASS' if passed else 'FAIL'}"])

    path = record_dir / "report.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, passed


def run_scenario(
    args: argparse.Namespace,
    uas: SipUas,
    log_path: Path,
    proc: subprocess.Popen,
    record_dir: Path,
    checks: List[Tuple[bool, str]],
) -> Tuple[Optional[RtpStats], Optional[Dict[str, float]], Optional[Path], Path]:
    """Один INVITE, разговор, BYE. Возвращает (RTP, анализ, запись клиента, WAV)."""
    caller_wav = record_dir / "caller_heard.wav"
    rtp: Optional[RtpStats] = None
    analysis: Optional[Dict[str, float]] = None
    recording: Optional[Path] = None

    ready = wait_for_log(log_path, "Многоканальный приём входящих", 10, proc)
    checks.append((ready, "Клиент перешёл в режим приёма входящих"))
    if not ready:
        return rtp, analysis, recording, caller_wav
    # recv-цикл pyVoIP стартует таймером через 1 с после REGISTER.
    time.sleep(1.2)

    rtp_sock = bind_udp(uas.my_ip)
    try:
        caller_rtp_port = rtp_sock.getsockname()[1]
        if args.caller_wav:
            pcm_s16, pcm_rate = read_wav(args.caller_wav)
            pcm_u8 = s16_to_u8(resample_s16(pcm_s16, pcm_rate, 8000))
            logger.info(
                "Вызывающий говорит WAV: %s (%.1f с)",
                args.caller_wav,
                len(pcm_u8) / 8000.0,
            )
        else:
            pcm_u8 = tone_u8(440.0, 1800, amplitude=0.5)
        tone_payloads = [
            ulaw_encode(frame)
            for frame in iter_frames(pcm_u8, FRAME_BYTES, SILENCE_BYTE)
        ]
        call_id = f"{uuid.uuid4().hex}@live-test"
        uas.send_invite(call_id, uuid.uuid4().hex[:8], caller_rtp_port)

        ringing = uas.ringing.wait(6)
        checks.append((ringing, "Получен 180 Ringing от клиента"))
        answered = uas.answered.wait(12)
        checks.append((answered, "Получен 200 OK на INVITE от клиента"))
        if answered and (not uas.client_ip or not uas.client_rtp_port):
            checks.append((False, "В SDP клиента есть c=IN IP4 и m=audio <port>"))
            answered = False
        else:
            checks.append(
                (
                    answered,
                    f"SDP клиента: {uas.client_ip}:{uas.client_rtp_port}",
                )
            )
        if not answered:
            return rtp, analysis, recording, caller_wav

        assert uas.client_ip is not None and uas.client_rtp_port is not None
        rtp = run_rtp(
            rtp_sock,
            (uas.client_ip, uas.client_rtp_port),
            args.duration,
            tone_payloads,
        )
        checks.append(
            (rtp.sent > 0, f"Вызывающий отправил {rtp.sent} RTP-пакетов")
        )
        checks.append(
            (
                rtp.received > 0,
                f"От клиента получено {rtp.received} RTP-пакетов "
                f"({rtp.duration_ms / 1000.0:.1f} с звука)",
            )
        )
        write_wav(caller_wav, u8_to_s16(bytes(rtp.pcm_u8)))
        analysis = analyze_pcm(rtp.pcm_u8)
        checks.append(
            (
                analysis["voiced"] > 0,
                "Вызывающий слышит клиента: "
                f"{int(analysis['voiced'])} звучных кадров, "
                f"RMS {analysis['peak']:.3f}",
            )
        )
    finally:
        rtp_sock.close()

    uas.send_bye(call_id)
    checks.append((uas.bye_ok.wait(6), "Получен 200 OK на BYE"))
    checks.append(
        (
            wait_for_log(log_path, "Звонок завершён", 6, proc),
            "Клиент завершил звонок (сессия закрыта)",
        )
    )
    checks.append(
        (
            wait_for_log(log_path, "Запись сохранена", 6, proc),
            "Recorder сохранил запись",
        )
    )
    recording = next(
        (p for p in sorted(record_dir.glob("*.wav")) if p.name != caller_wav.name),
        None,
    )
    checks.append(
        (
            bool(recording) and recording.stat().st_size > 44,
            f"Файл записи клиента: {recording.name if recording else 'нет'}",
        )
    )
    return rtp, analysis, recording, caller_wav


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.rtp_low > args.rtp_high:
        logger.error("--rtp-low не может быть больше --rtp-high")
        return 2
    if args.duration < 2:
        logger.error("--duration должен быть не меньше 2 с")
        return 2
    if args.client_sip_port == args.sip_port:
        logger.error("SIP-порты harness и клиента должны различаться")
        return 2
    if args.client_user == args.caller_user:
        logger.error("--client-user и --caller-user должны различаться")
        return 2
    if args.caller_wav and not Path(args.caller_wav).is_file():
        logger.error("--caller-wav не найден: %s", args.caller_wav)
        return 2

    record_dir = Path(args.record_dir).resolve()
    if record_dir.exists() and not args.keep:
        shutil.rmtree(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)

    my_ip = args.my_ip or detect_ip()
    if not my_ip:
        logger.error(
            "Не удалось определить IPv4 интерфейса. Укажите его через --my-ip "
            "(например, 192.168.1.130)."
        )
        return 2

    logger.info(
        "Конфигурация: harness %s:%d, клиент %s@%s:%d, RTP %d-%d",
        my_ip,
        args.sip_port,
        args.client_user,
        my_ip,
        args.client_sip_port,
        args.rtp_low,
        args.rtp_high,
    )

    sip_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sip_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sip_sock.bind((my_ip, args.sip_port))
    except OSError as exc:
        logger.error(
            "Не удалось занять %s:%d: %s (порт занят?)",
            my_ip,
            args.sip_port,
            exc,
        )
        sip_sock.close()
        return 2
    sip_sock.settimeout(0.2)

    uas = SipUas(
        sip_sock,
        my_ip,
        args.sip_port,
        args.client_user,
        args.client_sip_port,
        args.caller_user,
    )
    uas.start()

    log_path = record_dir / "client.log"
    checks: List[Tuple[bool, str]] = []
    proc: Optional[subprocess.Popen] = None
    log_handle = None
    rtp: Optional[RtpStats] = None
    analysis: Optional[Dict[str, float]] = None
    recording: Optional[Path] = None
    caller_wav = record_dir / "caller_heard.wav"

    try:
        env = os.environ.copy()
        env.update(
            {
                "SIP_SERVER": my_ip,
                "SIP_PORT": str(args.sip_port),
                "SIP_USER": args.client_user,
                "SIP_PASS": "live-test",
                "SIP_MY_IP": my_ip,
                "SIP_LOCAL_PORT": str(args.client_sip_port),
                "RTP_PORT_LOW": str(args.rtp_low),
                "RTP_PORT_HIGH": str(args.rtp_high),
                "MAX_INCOMING_LINES": "1",
            }
        )
        cmd = [
            sys.executable,
            str(ROOT / "main.py"),
            "-v",
            "incoming",
            "--bot",
            args.bot,
            "--record-dir",
            str(record_dir),
            "--max-lines",
            "1",
        ]
        if args.ui:
            cmd.extend(
                ["--ui", "--ui-host", "127.0.0.1", "--ui-port", str(args.ui_port)]
            )
            if args.no_asr:
                cmd.append("--no-asr")
        log_handle = log_path.open("w", encoding="utf-8")
        logger.info("Запуск клиента: %s", " ".join(cmd[1:]))
        if args.ui:
            logger.info("Веб-интерфейс: http://127.0.0.1:%d/", args.ui_port)
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        registered = uas.registered.wait(args.reg_timeout)
        checks.append(
            (registered, "Клиент зарегистрировался (REGISTER -> 200 OK)")
        )
        if not registered:
            logger.error(
                "Клиент не зарегистрировался за %.0f с", args.reg_timeout
            )
        else:
            logger.info("Клиент зарегистрирован, звоню один раз...")
            rtp, analysis, recording, caller_wav = run_scenario(
                args, uas, log_path, proc, record_dir, checks
            )
            if args.ui:
                if not args.no_asr:
                    time.sleep(1.5)
                ui_messages = fetch_ui_messages(args.ui_port)
                (record_dir / "ui_messages.json").write_text(
                    json.dumps(ui_messages, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                checks.append(
                    (
                        bool(ui_messages),
                        f"Веб-интерфейс отдал {len(ui_messages)} сообщений",
                    )
                )
                if args.ui_hold > 0:
                    logger.info(
                        "UI доступен %.0f с: http://127.0.0.1:%d/",
                        args.ui_hold,
                        args.ui_port,
                    )
                    time.sleep(args.ui_hold)
    finally:
        if proc is not None and proc.poll() is None:
            logger.info("Останавливаю клиента (SIGINT)...")
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.warning("Клиент не остановился за 15 с, terminate()")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        if log_handle is not None:
            log_handle.close()
        uas.stop()
        try:
            sip_sock.close()
        except OSError:
            pass

    log_text = read_log(log_path)
    checks.append(("Traceback" not in log_text, "В client.log нет Traceback"))
    checks.append(
        (
            "RTP Packet failed" not in log_text,
            "В client.log нет ошибок отправки RTP",
        )
    )

    report_path, passed = write_report(
        record_dir,
        my_ip,
        args,
        uas,
        rtp,
        analysis,
        recording,
        caller_wav,
        checks,
        log_text,
    )

    for ok, text in checks:
        if not ok:
            logger.error("FAIL: %s", text)
    logger.info("Отчёт: %s", report_path)
    print(f"ИТОГ: {'PASS' if passed else 'FAIL'}")
    print(f"Отчёт: {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

