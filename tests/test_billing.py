"""Тесты HTTP-клиента биллинга без реальной сети."""

from __future__ import annotations

import json
import urllib.error

import pytest

from sipbot.billing import BillingClient, BillingError


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def close(self) -> None:
        pass


def test_billing_posts_user_id_and_bearer_token(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None, context=None):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse(
            json.dumps(
                {"found": True, "user": {"id": "20", "extra_account": 429.56}}
            ).encode("utf-8")
        )

    monkeypatch.setattr("sipbot.billing.urllib.request.urlopen", fake_urlopen)
    client = BillingClient("https://billing.example/api", "secret", timeout=5)

    data = client.get_account(" 20 ")

    assert data["found"] is True
    assert data["user"]["extra_account"] == 429.56
    assert captured["url"] == "https://billing.example/api"
    assert captured["body"] == {"userId": "20"}
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["timeout"] == 5


def test_billing_sslip_falls_back_to_ip_but_keeps_host_header(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None, context=None):
        captured["url"] = request.full_url
        captured["host"] = request.get_header("Host")
        return FakeResponse(b'{"found": false}')

    monkeypatch.setattr("sipbot.billing.urllib.request.urlopen", fake_urlopen)
    client = BillingClient(
        "http://maxbot.91.195.230.18.sslip.io/integration/subscriber/by-id",
        "secret",
    )

    assert client.get_account("20") == {"found": False}
    assert captured["url"] == (
        "http://91.195.230.18/integration/subscriber/by-id"
    )
    assert captured["host"] == "maxbot.91.195.230.18.sslip.io"


def test_billing_rejects_non_digit_account_before_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("сеть не должна вызываться")

    monkeypatch.setattr("sipbot.billing.urllib.request.urlopen", fail)
    client = BillingClient("https://billing.example/api", "secret")

    with pytest.raises(BillingError, match="Некорректный"):
        client.get_account("20a")


def test_billing_without_config_raises():
    client = BillingClient()

    assert client.configured is False
    with pytest.raises(BillingError, match="не настроен"):
        client.get_account("20")


def test_billing_wraps_http_error(monkeypatch):
    def fake_urlopen(request, timeout=None, context=None):
        raise urllib.error.HTTPError(
            request.full_url, 401, "Unauthorized", {}, FakeResponse(b"denied")
        )

    monkeypatch.setattr("sipbot.billing.urllib.request.urlopen", fake_urlopen)
    client = BillingClient("https://billing.example/api", "bad")

    with pytest.raises(BillingError, match="HTTP 401"):
        client.get_account("20")


def test_billing_from_env_requires_url_and_token():
    assert BillingClient.from_env({}) is None
    assert BillingClient.from_env({"BILLING_BASE_URL": "https://x"}) is None

    client = BillingClient.from_env(
        {
            "BILLING_BASE_URL": "https://x",
            "BILLING_TOKEN": "token",
            "BILLING_TIMEOUT": "3.5",
        }
    )
    assert client is not None
    assert client.base_url == "https://x"
    assert client.token == "token"
    assert client.timeout == 3.5
