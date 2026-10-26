"""Тесты точного ожидания дедлайна."""

from sipbot.timing import now, wait_until


def test_wait_until_does_not_wake_early():
    deadline = now() + 0.02
    wait_until(deadline)
    assert now() >= deadline


def test_wait_until_returns_immediately_for_past_deadline():
    started = now()
    wait_until(started - 0.01)
    assert now() - started < 0.01
