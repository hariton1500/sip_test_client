"""Точное ожидание дедлайна для медиа-циклов.

На Windows ``time.monotonic()`` — это ``GetTickCount64`` с разрешением
15.6 мс, а ``time.sleep(0.02)`` просыпается через 16 или 32 мс. Для RTP это
означает то недосыл, то пересыл кадров: pyVoIP вставляет кадры тишины, и речь
звучит рвано. Поэтому все дедлайны считаем по ``time.perf_counter()``
(``QueryPerformanceCounter``, шаг ~100 нс) и ждём «сон до дедлайна минус
1.5 мс + короткий активный спин».
"""

from __future__ import annotations

import time

#: За сколько до дедлайна переходить из sleep в активное ожидание.
SPIN_MARGIN = 0.0015


def now() -> float:
    """Монотонное время высокого разрешения для дедлайнов."""
    return time.perf_counter()


def wait_until(deadline: float) -> float:
    """Дождаться ``deadline`` (в шкале :func:`now`) и вернуть опоздание, с."""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= SPIN_MARGIN:
            break
        time.sleep(max(0.0, remaining - SPIN_MARGIN))
    while time.perf_counter() < deadline:
        pass
    return max(0.0, time.perf_counter() - deadline)


__all__ = ["SPIN_MARGIN", "now", "wait_until"]
