"""Заглушка клиента биллинга Евпанет.

В будущем здесь будет реальный HTTP-запрос к биллинговой системе.
Сейчас метод возвращает тестовые данные для отладки диалогового сценария.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BillingError(Exception):
    """Ошибка при обращении к биллингу."""


class BillingClient:
    """Клиент для получения данных абонента по лицевому счёту."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = base_url

    def get_account(self, account_id: str) -> Dict[str, Any]:
        """Вернуть данные абонента по лицевому счёту.

        TODO: заменить на реальный HTTP-запрос к API биллинга Евпанет.
        """
        if not account_id or not account_id.strip().isdigit():
            raise BillingError("Некорректный номер лицевого счёта")

        logger.debug("BILLING: запрос данных по счёту %s", account_id)

        # Заглушка с детерминированными тестовыми данными.
        return {
            "account_id": account_id,
            "balance": 1240.50,
            "status": "active",
            "tariff": "Домашний Оптима 100",
            "services": ["Интернет 100 Мбит/с", "IPTV", "Телефония"],
            "last_payment": "2026-10-20",
            "last_payment_amount": 1500.00,
            "support_pin": "4821",
        }
