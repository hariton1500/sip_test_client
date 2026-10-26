"""HTTP-клиент биллинга Евпанет.

API: ``POST {BILLING_BASE_URL}`` с ``Authorization: Bearer <BILLING_TOKEN>``
и телом ``{"userId": "<номер лицевого счёта>"}``. Ответ — JSON с ``found`` и
объектом ``user`` (см. ``prompts/billing_account_data.md``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)


class BillingError(Exception):
    """Ошибка при обращении к биллингу."""


class BillingClient:
    """Клиент для получения данных абонента по лицевому счёту."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: str = "",
        timeout: float = 10.0,
        ssl_context: Optional[ssl.SSLContext] = None,
    ) -> None:
        self.base_url = (base_url or "").strip()
        self.token = (token or "").strip()
        self.timeout = timeout
        self._ssl_context = ssl_context or ssl.create_default_context()

    @classmethod
    def from_env(
        cls, env: Optional[Mapping[str, str]] = None
    ) -> Optional["BillingClient"]:
        """Создать клиент из окружения; None, если биллинг не настроен."""
        source = env if env is not None else os.environ
        base_url = str(source.get("BILLING_BASE_URL", "") or "").strip()
        token = str(source.get("BILLING_TOKEN", "") or "").strip()
        if not base_url or not token:
            return None
        try:
            timeout = float(source.get("BILLING_TIMEOUT", "10") or "10")
        except ValueError:
            logger.warning(
                "Некорректный BILLING_TIMEOUT=%r, использую 10 с",
                source.get("BILLING_TIMEOUT"),
            )
            timeout = 10.0
        return cls(base_url=base_url, token=token, timeout=max(0.5, timeout))

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    def _request_target(self) -> tuple[str, Optional[str]]:
        """URL и необязательный Host для sslip.io без рабочего DNS.

        ``maxbot.91.195.230.18.sslip.io`` по замыслу резолвится в
        ``91.195.230.18``. Если локальный DNS не умеет sslip.io, для HTTP
        подключаемся прямо к IP, но сохраняем исходный Host: виртуальный хост
        nginx выбирается именно по нему.
        """
        parsed = urllib.parse.urlsplit(self.base_url)
        host = parsed.hostname or ""
        if parsed.scheme != "http":
            return self.base_url, None
        match = re.search(
            r"(?:^|\.)(\d{1,3}(?:\.\d{1,3}){3})\.sslip\.io$", host
        )
        if match is None:
            return self.base_url, None
        netloc = match.group(1)
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        target = urllib.parse.urlunsplit(
            (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
        )
        host_header = f"{host}:{parsed.port}" if parsed.port else host
        logger.debug("BILLING: DNS fallback %s -> %s", host, netloc)
        return target, host_header

    def get_account(self, account_id: str) -> Dict[str, Any]:
        """Запросить данные абонента по номеру лицевого счёта."""
        account_id = (account_id or "").strip()
        if not account_id.isdigit():
            raise BillingError("Некорректный номер лицевого счёта")
        if not self.configured:
            raise BillingError(
                "Биллинг не настроен: задайте BILLING_BASE_URL и BILLING_TOKEN"
            )

        payload = json.dumps(
            {"userId": account_id}, ensure_ascii=False
        ).encode("utf-8")
        target, host_header = self._request_target()
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if host_header:
            headers["Host"] = host_header
        request = urllib.request.Request(
            target,
            data=payload,
            headers=headers,
            method="POST",
        )
        logger.info("BILLING: запрос данных по счёту %s", account_id)
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
                context=self._ssl_context,
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            raise BillingError(
                f"HTTP {exc.code} {exc.reason}: {body}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - сеть/таймаут/SSL
            raise BillingError(f"Ошибка запроса к биллингу: {exc}") from exc

        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BillingError(f"Биллинг вернул некорректный JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise BillingError("Биллинг вернул JSON не-объект")
        logger.info(
            "BILLING: счёт %s, found=%s", account_id, data.get("found")
        )
        return data
