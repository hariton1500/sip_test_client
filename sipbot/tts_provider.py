"""Выбор TTS-провайдера: облачный AiClient, локальный kokoro-ru или гибрид.

``TTS_PROVIDER=cloud`` (по умолчанию) сохраняет прежнее поведение.
``TTS_PROVIDER=kokoro`` включает локальную модель и падает на старте, если
она недоступна. ``TTS_PROVIDER=auto`` пробует локальную модель, а при ошибке
синтеза откатывается на облачный TTS.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Mapping, Optional, Union

from .tts import TtsService
from .tts_kokoro import KokoroTts

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

PROVIDER_CLOUD = "cloud"
PROVIDER_KOKORO = "kokoro"
PROVIDER_AUTO = "auto"
PROVIDERS = (PROVIDER_CLOUD, PROVIDER_KOKORO, PROVIDER_AUTO)


class FallbackTts:
    """Пробует локальный TTS, при исключении — облачный.

    Публичные ``tts_*`` атрибуты берутся у основного провайдера, чтобы ключ
    дискового кэша оставался стабильным. Если локальный синтез упал и ответ
    сохранил облачный провайдер, этот WAV тоже попадёт в кэш — при следующем
    звонке он проиграется без обращения к сети.
    """

    def __init__(self, primary, fallback) -> None:
        self.primary = primary
        self.fallback = fallback
        self.tts_model = str(getattr(primary, "tts_model", "local"))
        self.tts_voice = str(getattr(primary, "tts_voice", ""))
        self.tts_sample_rate = int(
            getattr(primary, "tts_sample_rate", 24000) or 24000
        )
        self.tts_response_format = str(
            getattr(primary, "tts_response_format", "wav")
        )

    def synthesize(self, text: str) -> bytes:
        try:
            return self.primary.synthesize(text)
        except Exception as exc:  # noqa: BLE001 - откат не должен рвать звонок
            if self.fallback is None:
                raise
            logger.warning(
                "Локальный TTS не сработал (%s); синтезирую облачным", exc
            )
            return self.fallback.synthesize(text)

    def warmup(self) -> None:
        warmup = getattr(self.primary, "warmup", None)
        if callable(warmup):
            warmup()

    def close(self) -> None:
        for provider in (self.primary, self.fallback):
            close = getattr(provider, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    logger.exception("Ошибка закрытия TTS-провайдера")


def build_tts_provider(
    ai, environ: Optional[Mapping[str, str]] = None
):
    """Вернуть объект с методом ``synthesize`` для ``TtsService``."""
    source: Mapping[str, str] = os.environ if environ is None else environ
    provider = str(source.get("TTS_PROVIDER", PROVIDER_CLOUD) or "").strip().lower()
    if not provider:
        provider = PROVIDER_CLOUD
    if provider == "local":
        provider = PROVIDER_KOKORO
    if provider not in PROVIDERS:
        raise ValueError(
            f"TTS_PROVIDER={provider!r} не поддерживается; "
            f"допустимо: {', '.join(PROVIDERS)}"
        )

    if provider == PROVIDER_CLOUD:
        if ai is None:
            raise ValueError("TTS_PROVIDER=cloud требует OPENAI_BASE_URL/API_KEY")
        return ai

    try:
        local = KokoroTts.from_env(source)
    except Exception as exc:  # noqa: BLE001 - auto должен пережить любую ошибку
        if provider == PROVIDER_KOKORO:
            raise ValueError(
                f"kokoro-ru не запустился: {exc}. Установите "
                "requirements-kokoro.txt или переключитесь на TTS_PROVIDER=cloud"
            ) from exc
        logger.warning(
            "kokoro-ru недоступен, использую облачный TTS: %s", exc
        )
        return ai

    if provider == PROVIDER_KOKORO or ai is None:
        return local
    logger.info("TTS: локальный kokoro-ru с откатом на облачный")
    return FallbackTts(local, ai)


def build_tts_service(
    ai,
    cache_dir: Optional[PathLike] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> TtsService:
    """Собрать ``TtsService`` с выбранным провайдером."""
    provider = build_tts_provider(ai, environ)
    logger.info(
        "TTS-провайдер: %s/%s",
        getattr(provider, "tts_model", "?"),
        getattr(provider, "tts_voice", "?"),
    )
    return TtsService(provider, cache_dir=cache_dir)


__all__ = [
    "FallbackTts",
    "PROVIDERS",
    "build_tts_provider",
    "build_tts_service",
]
