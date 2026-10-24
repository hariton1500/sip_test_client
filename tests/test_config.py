"""Тесты конфигурации: разбор .env, валидация, маскирование секретов."""

import pytest

from sipbot.config import Settings, _read_env_file

BASE_ENV = {
    "SIP_SERVER": "pbx.local",
    "SIP_USER": "1000",
    "SIP_PASS": "supersecret",
    "SIP_MY_IP": "192.168.1.50",
}


def make(**overrides) -> Settings:
    return Settings.from_env(environ={**BASE_ENV, **overrides})


def test_defaults_and_validation():
    settings = make()
    assert settings.server == "pbx.local"
    assert settings.port == 5060
    assert settings.sip_port == 5060
    assert settings.rtp_port_low == 10000
    assert settings.rtp_port_high == 10100
    settings.validate()


def test_reads_optional_values():
    settings = make(
        SIP_PORT="5070",
        SIP_LOCAL_PORT="5080",
        RTP_PORT_LOW="20000",
        RTP_PORT_HIGH="20100",
    )
    assert (settings.port, settings.sip_port) == (5070, 5080)
    assert (settings.rtp_port_low, settings.rtp_port_high) == (20000, 20100)


def test_missing_my_ip_rejected():
    settings = make(SIP_MY_IP="")
    with pytest.raises(ValueError, match="SIP_MY_IP"):
        settings.validate()


def test_zero_ip_rejected():
    with pytest.raises(ValueError, match="0.0.0.0"):
        make(SIP_MY_IP="0.0.0.0").validate()


def test_loopback_ip_rejected():
    with pytest.raises(ValueError, match="loopback"):
        make(SIP_MY_IP="127.0.0.1").validate()


def test_non_ip_value_rejected():
    with pytest.raises(ValueError, match="IP-адресом"):
        make(SIP_MY_IP="my-host").validate()


def test_ipv6_rejected():
    with pytest.raises(ValueError, match="IPv4"):
        make(SIP_MY_IP="::1").validate()


def test_missing_credentials_rejected():
    with pytest.raises(ValueError) as exc:
        make(SIP_SERVER="", SIP_USER="", SIP_PASS="").validate()
    message = str(exc.value)
    assert "SIP_SERVER" in message
    assert "SIP_USER" in message
    assert "SIP_PASS" in message


def test_bad_ports_rejected():
    with pytest.raises(ValueError, match="SIP_PORT"):
        make(SIP_PORT="70000").validate()
    with pytest.raises(ValueError, match="RTP_PORT_LOW"):
        make(RTP_PORT_LOW="0").validate()


def test_rtp_range_order_rejected():
    with pytest.raises(ValueError, match="RTP_PORT_LOW"):
        make(RTP_PORT_LOW="10200", RTP_PORT_HIGH="10100").validate()


def test_describe_hides_password():
    description = make().describe()
    assert "supersecret" not in description
    assert "pbx.local" in description
    assert "192.168.1.50" in description


def test_read_env_file_fallback(tmp_path):
    path = tmp_path / "custom.env"
    path.write_text(
        "# комментарий\n"
        "\n"
        "SIP_SERVER=1.2.3.4\n"
        'SIP_PASS="secret"\n'
        "SIP_USER='1000'\n",
        encoding="utf-8",
    )
    assert _read_env_file(path) == {
        "SIP_SERVER": "1.2.3.4",
        "SIP_PASS": "secret",
        "SIP_USER": "1000",
    }
    assert _read_env_file(tmp_path / "missing.env") == {}
