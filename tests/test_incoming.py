"""Тесты разбора Caller ID из входящего INVITE."""

from __future__ import annotations

from types import SimpleNamespace

from sipbot.incoming import caller_info


def _call(number: str, caller: str, call_id: str = "call-1"):
    request = SimpleNamespace(headers={"From": {"number": number, "caller": caller}})
    return SimpleNamespace(request=request, call_id=call_id)


def test_caller_info_prefers_phone_from_display_name():
    info = caller_info(_call("508", ' +79780481885" '))

    assert info.number == "+79780481885"
    assert info.display == "+79780481885"
    assert info.sip_user == "508"


def test_caller_info_keeps_sip_user_when_display_is_a_name():
    info = caller_info(_call("508", '"Тест"'))

    assert info.number == "508"
    assert info.display == "Тест"
    assert info.sip_user == "508"


def test_caller_info_handles_missing_from_details():
    info = caller_info(SimpleNamespace(request=None, call_id=None))

    assert info.number == "unknown"
    assert info.display == ""
    assert info.call_id == "?"
