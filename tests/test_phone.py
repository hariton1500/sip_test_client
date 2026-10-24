"""Тесты нормализации цели звонка (без обращения к сети)."""

import pytest

from sipbot.phone import normalize_target


@pytest.mark.parametrize(
    "target,expected",
    [
        ("1001", "1001"),
        (" 1001 ", "1001"),
        ("sip:1001", "1001"),
        ("sip:1001@pbx.local", "1001"),
        ("sip:1001@pbx.local:5060", "1001"),
        ("sips:1001@pbx.local", "1001"),
        ("tel:+79001234567", "+79001234567"),
    ],
)
def test_normalize_target(target, expected):
    assert normalize_target(target) == expected


def test_normalize_target_warns_on_foreign_host(caplog):
    with caplog.at_level("WARNING"):
        assert normalize_target("sip:1001@other.host", "pbx.local") == "1001"
    assert "other.host" in caplog.text


def test_normalize_target_no_warning_for_same_host(caplog):
    with caplog.at_level("WARNING"):
        normalize_target("sip:1001@pbx.local:5060", "pbx.local")
    assert caplog.text == ""


@pytest.mark.parametrize("target", ["", "   ", "sip:", "sip:@pbx.local"])
def test_normalize_target_rejects_empty(target):
    with pytest.raises(ValueError):
        normalize_target(target)
