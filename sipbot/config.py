"""Сеть/конфигурация SIP-клиента."""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Union

PathLike = Union[str, Path]


def _read_env_file(path: PathLike) -> dict:
    """Минимальный парсер .env (используется, если python-dotenv не установлен)."""
    values = {}
    file_path = Path(path)
    if not file_path.exists():
        return values
    for raw in file_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _load_dotenv(path: Optional[PathLike]) -> None:
    if not path:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path)
        return
    except ImportError:
        pass
    for key, value in _read_env_file(path).items():
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    """Параметры подключения к SIP-серверу (АТС)."""

    server: str
    username: str
    password: str
    my_ip: str
    port: int = 5060
    sip_port: int = 5060
    rtp_port_low: int = 10000
    rtp_port_high: int = 10100
    max_incoming_lines: int = 16

    @classmethod
    def from_env(
        cls,
        env_file: Optional[PathLike] = ".env",
        environ: Optional[Mapping[str, str]] = None,
    ) -> "Settings":
        if environ is None:
            _load_dotenv(env_file)
            environ = os.environ

        def get(name: str, default: str = "") -> str:
            return str(environ.get(name, default) or default).strip()

        def get_int(name: str, default: int) -> int:
            raw = get(name)
            return int(raw) if raw else default

        return cls(
            server=get("SIP_SERVER"),
            username=get("SIP_USER"),
            password=get("SIP_PASS"),
            my_ip=get("SIP_MY_IP"),
            port=get_int("SIP_PORT", 5060),
            sip_port=get_int("SIP_LOCAL_PORT", 5060),
            rtp_port_low=get_int("RTP_PORT_LOW", 10000),
            rtp_port_high=get_int("RTP_PORT_HIGH", 10100),
            max_incoming_lines=get_int("MAX_INCOMING_LINES", 16),
        )

    def validate(self) -> None:
        errors = []

        if not self.server:
            errors.append("SIP_SERVER не задан")
        if not self.username:
            errors.append("SIP_USER не задан")
        if not self.password:
            errors.append("SIP_PASS не задан")
        for name, value in (("SIP_PORT", self.port), ("SIP_LOCAL_PORT", self.sip_port)):
            if not 0 < value < 65536:
                errors.append(f"{name} вне диапазона 1..65535: {value}")
        for name, value in (
            ("RTP_PORT_LOW", self.rtp_port_low),
            ("RTP_PORT_HIGH", self.rtp_port_high),
        ):
            if not 0 < value < 65536:
                errors.append(f"{name} вне диапазона 1..65535: {value}")
        if self.rtp_port_low > self.rtp_port_high:
            errors.append(
                f"RTP_PORT_LOW ({self.rtp_port_low}) > RTP_PORT_HIGH "
                f"({self.rtp_port_high})"
            )

        available_ports = self.rtp_port_high - self.rtp_port_low + 1
        if available_ports < self.max_incoming_lines:
            errors.append(
                f"RTP-диапазон ({available_ports} портов) меньше "
                f"MAX_INCOMING_LINES ({self.max_incoming_lines})"
            )
        if self.max_incoming_lines < 1:
            errors.append("MAX_INCOMING_LINES должен быть >= 1")

        if not self.my_ip:
            errors.append(
                "SIP_MY_IP не задан: укажите локальный IPv4 этой машины, "
                "иначе в SDP уйдёт 0.0.0.0 и звук пойдёт в одну сторону"
            )
        else:
            try:
                ip = ipaddress.ip_address(self.my_ip)
            except ValueError:
                errors.append(f"SIP_MY_IP не является IP-адресом: {self.my_ip}")
            else:
                if ip.version != 4:
                    errors.append("pyVoIP работает только с IPv4")
                elif ip.is_unspecified:
                    errors.append(
                        "SIP_MY_IP не может быть 0.0.0.0 (нужен реальный адрес "
                        "интерфейса)"
                    )
                elif ip.is_loopback:
                    errors.append(
                        "SIP_MY_IP не может быть loopback-адресом (нужен адрес "
                        "интерфейса, доступный АТС)"
                    )

        if errors:
            raise ValueError("Некорректная конфигурация: " + "; ".join(errors))

    def describe(self) -> str:
        return (
            f"server={self.server}:{self.port} user={self.username} "
            f"my_ip={self.my_ip} sip_port={self.sip_port} "
            f"rtp={self.rtp_port_low}-{self.rtp_port_high} "
            f"max_lines={self.max_incoming_lines}"
        )


__all__ = ["Settings"]
