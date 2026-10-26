"""OpenAI-совместимый клиент для ASR/LLM/TTS.

Работает через стандартную библиотеку (``http.client`` + ``ssl``), чтобы не
зависеть от requests/httpx. Поддерживает любые OpenAI-совместимые API, в том
числе AiTunnel.

HTTP-соединение переиспользуется в пределах потока: в телефонном пайплайне
на одну реплику приходится три запроса (ASR, LLM, TTS), и новый TLS-handshake
на каждый из них заметно добавляет к time-to-first-audio.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import re
import socket
import ssl
import threading
import urllib.parse
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class AiClientError(Exception):
    """Ошибка при обращении к AI API."""


@dataclass(frozen=True)
class ChatResponse:
    """Ответ LLM: текст и/или tool_calls."""

    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def _default_ssl_context() -> ssl.SSLContext:
    """SSL-контекст с системными/сертификатными корнями.

    В отдельных сборках Python (например, на macOS) ``ssl`` не находит
    CA-файл по умолчанию. Тогда подхватываем certifi или стандартные пути,
    чтобы HTTPS-запросы к AI API не падали с CERTIFICATE_VERIFY_FAILED.
    """
    try:
        import certifi  # type: ignore
    except ImportError:
        pass
    else:
        return ssl.create_default_context(cafile=certifi.where())

    context = ssl.create_default_context()
    if context.cert_store_stats().get("x509_ca", 0) == 0:
        for path in (
            os.environ.get("SSL_CERT_FILE", ""),
            "/etc/ssl/cert.pem",
            "/etc/pki/tls/certs/ca-bundle.crt",
            "/etc/ssl/certs/ca-certificates.crt",
        ):
            if path and os.path.isfile(path):
                context.load_verify_locations(cafile=path)
                break
    return context


class AiClient:
    """Клиент ASR/LLM/TTS через OpenAI-совместимое API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        asr_model: str = "whisper-large-v3-turbo",
        llm_model: str = "gemini-3.1-flash-lite",
        tts_model: str = "gemini-3.8-flash-lite-tts",
        tts_voice: str = "Zephyr",
        tts_response_format: str = "pcm",
        tts_sample_rate: int = 24000,
        tts_timeout: float = 20.0,
        asr_language: str = "ru",
        timeout: float = 30.0,
        # Быстрая модель без reasoning: 150 токенов хватает на 1-2 коротких
        # предложения. Для reasoning-моделей лимит нужно поднимать.
        llm_max_tokens: int = 150,
        llm_temperature: float = 0.3,
        # Маршрутизация провайдеров AITUNNEL: мягкое предпочтение быстрых
        # провайдеров без потери запасных (см. docs/provider).
        llm_provider: Optional[Dict[str, Any]] = None,
        llm_service_tier: str = "",
        llm_fallback_models: Sequence[str] = (),
        llm_reasoning_effort: str = "",
        ssl_context: Optional[ssl.SSLContext] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.asr_model = asr_model
        self.llm_model = llm_model
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.tts_response_format = tts_response_format
        self.tts_sample_rate = tts_sample_rate
        self.tts_timeout = tts_timeout
        self.asr_language = asr_language
        self.timeout = timeout
        self.llm_max_tokens = max(1, int(llm_max_tokens))
        self.llm_temperature = min(2.0, max(0.0, float(llm_temperature)))
        self.llm_provider = dict(llm_provider) if llm_provider else None
        self.llm_service_tier = (llm_service_tier or "").strip().lower()
        # AITUNNEL принимает не больше 5 моделей в models, включая основную.
        self.llm_fallback_models = [
            str(model).strip()
            for model in llm_fallback_models
            if str(model).strip()
        ][:4]
        self.llm_reasoning_effort = (llm_reasoning_effort or "").strip().lower()
        self._ssl_context = ssl_context or _default_ssl_context()

        parsed = urllib.parse.urlsplit(self.base_url)
        self._scheme = (parsed.scheme or "https").lower()
        self._host = parsed.hostname or ""
        self._port = parsed.port
        self._base_path = parsed.path.rstrip("/")
        # Соединения живут в потоке: AiClient используется из нескольких
        # рабочих потоков диалога, а http.client не потокобезопасен.
        self._local = threading.local()

    @classmethod
    def from_env(
        cls, env: Optional[Mapping[str, str]] = None
    ) -> Optional["AiClient"]:
        """Создаёт клиент из переменных окружения или возвращает None.

        Поддерживаются как ``OPENAI_*``, так и короткие ``AI_*`` имена.
        Если URL или ключ не заданы, ASR/LLM/TTS просто отключаются.
        """
        source = env if env is not None else os.environ

        def read(*names: str, default: str = "") -> str:
            for name in names:
                value = source.get(name)
                if value:
                    return value
            return default

        def read_int(name: str, default: int) -> int:
            try:
                return int(read(name, default=str(default)))
            except ValueError:
                logger.warning("Некорректное %s=%r, использую %d", name, source.get(name), default)
                return default

        def read_float(name: str, default: float) -> float:
            try:
                return float(read(name, default=str(default)))
            except ValueError:
                logger.warning("Некорректное %s=%r, использую %.2f", name, source.get(name), default)
                return default

        def read_list(name: str) -> List[str]:
            raw = read(name)
            return [part.strip() for part in re.split(r"[,;]", raw) if part.strip()]

        provider: Dict[str, Any] = {}
        latency_p90 = read_float("LLM_PROVIDER_MAX_LATENCY_P90", 1.2)
        if latency_p90 > 0:
            provider["preferred_max_latency"] = {"p90": latency_p90}
        min_throughput = read_float("LLM_PROVIDER_MIN_THROUGHPUT", 0.0)
        if min_throughput > 0:
            provider["preferred_min_throughput"] = min_throughput
        provider_sort = read("LLM_PROVIDER_SORT").strip().lower()
        if provider_sort in ("price", "throughput", "latency"):
            provider["sort"] = provider_sort
        provider_only = read_list("LLM_PROVIDER_ONLY")
        if provider_only:
            provider["only"] = provider_only
        provider_ignore = read_list("LLM_PROVIDER_IGNORE")
        if provider_ignore:
            provider["ignore"] = provider_ignore

        base_url = read("OPENAI_BASE_URL", "AI_BASE_URL")
        api_key = read("OPENAI_API_KEY", "AI_API_KEY")
        if not base_url or not api_key:
            return None

        return cls(
            base_url=base_url,
            api_key=api_key,
            asr_model=read(
                "ASR_MODEL", "OPENAI_ASR_MODEL", default="whisper-large-v3-turbo"
            ),
            llm_model=read(
                "LLM_MODEL", "OPENAI_LLM_MODEL", default="gemini-3.1-flash-lite"
            ),
            tts_model=read(
                "TTS_MODEL", "OPENAI_TTS_MODEL", default="gemini-3.8-flash-lite-tts"
            ),
            tts_voice=read("TTS_VOICE", "OPENAI_TTS_VOICE", default="Zephyr"),
            tts_response_format=read(
                "TTS_RESPONSE_FORMAT", "OPENAI_TTS_RESPONSE_FORMAT", default="pcm"
            ),
            tts_sample_rate=read_int("TTS_SAMPLE_RATE", 24000),
            tts_timeout=read_float("TTS_TIMEOUT", 20.0),
            asr_language=read("ASR_LANGUAGE", "OPENAI_ASR_LANGUAGE", default="ru"),
            timeout=read_float("AI_TIMEOUT", 30.0),
            llm_max_tokens=read_int("LLM_MAX_TOKENS", 150),
            llm_temperature=read_float("LLM_TEMPERATURE", 0.3),
            llm_provider=provider or None,
            llm_service_tier=read("LLM_SERVICE_TIER"),
            llm_fallback_models=read_list("LLM_FALLBACK_MODELS"),
            llm_reasoning_effort=read("LLM_REASONING_EFFORT"),
        )

    def _new_connection(self, timeout: float) -> http.client.HTTPConnection:
        if self._scheme == "https":
            return http.client.HTTPSConnection(
                self._host,
                self._port,
                timeout=timeout,
                context=self._ssl_context,
            )
        return http.client.HTTPConnection(self._host, self._port, timeout=timeout)

    def _connection(self, timeout: float) -> http.client.HTTPConnection:
        """Вернуть keep-alive соединение текущего потока."""
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = self._new_connection(timeout)
            self._local.connection = connection
        else:
            connection.timeout = timeout
            if connection.sock is not None:
                connection.sock.settimeout(timeout)
        return connection

    def _drop_connection(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            try:
                connection.close()
            except Exception:  # noqa: BLE001 - закрытие не должно ломать запрос
                pass
            self._local.connection = None

    def _raw_request(
        self,
        method: str,
        endpoint: str,
        body: Optional[bytes],
        headers: Dict[str, str],
        timeout: float,
    ) -> Tuple[int, bytes, str]:
        """Выполнить HTTP-запрос, переиспользуя соединение потока.

        Повторяем ровно один раз и только если соединение было взято из пула
        (keep-alive): типичная причина ошибки — сервер уже закрыл простаивающее
        соединение. Свежее соединение не повторяем, чтобы не дублировать
        оплачиваемый запрос к LLM.
        """
        path = f"{self._base_path}{endpoint}" or "/"
        for attempt in (1, 2):
            connection = self._connection(timeout)
            reused = connection.sock is not None
            try:
                connection.request(method, path, body=body, headers=headers)
                response = connection.getresponse()
                raw = response.read()
                status = response.status
                reason = response.reason
                if response.will_close:
                    self._drop_connection()
                return status, raw, reason
            except (http.client.HTTPException, OSError) as exc:
                self._drop_connection()
                # Таймаут мог наступить уже после отправки запроса: повтор
                # рискует оплатить его дважды.
                retryable = not isinstance(exc, (socket.timeout, TimeoutError))
                if not reused or attempt == 2 or not retryable:
                    raise
                logger.debug(
                    "Повтор %s %s: простаивающее соединение закрыто (%s)",
                    method,
                    endpoint,
                    exc,
                )
        raise AiClientError(f"Не удалось выполнить {method} {endpoint}")

    def _request(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        data: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        req_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        if headers:
            req_headers.update(headers)

        body: Optional[bytes] = data
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")

        logger.debug("AI API %s %s", method, endpoint)
        try:
            status, raw, reason = self._raw_request(
                method, endpoint, body, req_headers, self.timeout
            )
        except Exception as exc:
            raise AiClientError(f"Ошибка запроса к {url}: {exc}") from exc

        if status >= 400:
            error_body = raw.decode("utf-8", errors="replace")[:500]
            raise AiClientError(f"HTTP {status} {reason}: {error_body}")

        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise AiClientError(f"Некорректный JSON от {endpoint}: {exc}") from exc

    def transcribe(
        self,
        wav_bytes: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> str:
        """Распознать речь из WAV s16le. Возвращает текст."""
        boundary = f"----FormBoundary{uuid.uuid4().hex}"
        fields: List[tuple] = [("model", self.asr_model)]
        language = language if language is not None else self.asr_language
        if language:
            fields.append(("language", language))
        if prompt:
            fields.append(("prompt", prompt))
        body = self._build_multipart(
            fields,
            file_name="utterance.wav",
            file_content_type="audio/wav",
            file_data=wav_bytes,
            boundary=boundary,
        )
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        result = self._request(
            "POST",
            "/audio/transcriptions",
            data=body,
            headers=headers,
        )
        text = result.get("text") or ""
        logger.debug("ASR: %r", text)
        return text

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatResponse:
        """Отправить диалог в LLM. Возвращает текст и/или tool_calls."""
        payload: Dict[str, Any] = {
            "model": self.llm_model,
            "messages": messages,
            "max_tokens": self.llm_max_tokens,
            "temperature": self.llm_temperature,
        }
        if self.llm_fallback_models:
            # AITUNNEL: model идёт первым, models — очередь запасных.
            payload["models"] = [self.llm_model, *self.llm_fallback_models]
        if self.llm_provider:
            payload["provider"] = self.llm_provider
        if self.llm_service_tier:
            payload["service_tier"] = self.llm_service_tier
        if self.llm_reasoning_effort:
            payload["reasoning_effort"] = self.llm_reasoning_effort
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        result = self._request("POST", "/chat/completions", payload=payload)
        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls")
        usage = result.get("usage")
        finish_reason = choice.get("finish_reason")
        served_model = result.get("model")
        if served_model and served_model != self.llm_model:
            logger.info("LLM ответила запасная модель: %s", served_model)
        if usage:
            logger.info(
                "LLM usage: prompt=%s completion=%s total=%s finish=%s tier=%s",
                usage.get("prompt_tokens", "?"),
                usage.get("completion_tokens", "?"),
                usage.get("total_tokens", "?"),
                finish_reason or "?",
                result.get("service_tier") or "-",
            )
        logger.debug("LLM: content=%r tool_calls=%s", content, bool(tool_calls))
        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
        )

    def synthesize(self, text: str) -> bytes:
        """Синтезировать речь.

        Формат ответа задаёт ``TTS_RESPONSE_FORMAT`` (по умолчанию ``pcm``:
        raw s16le mono с частотой ``TTS_SAMPLE_RATE``). Провайдер может вернуть
        WAV независимо от запроса — ``TtsService`` определяет это по заголовку.
        """
        payload = {
            "model": self.tts_model,
            "voice": self.tts_voice,
            "input": text,
            "response_format": self.tts_response_format,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "audio/wav,audio/*",
        }
        try:
            status, audio, reason = self._raw_request(
                "POST", "/audio/speech", body, headers, self.tts_timeout
            )
        except Exception as exc:
            raise AiClientError(f"Ошибка TTS: {exc}") from exc

        if status >= 400:
            error_body = audio.decode("utf-8", errors="replace")[:500]
            raise AiClientError(f"HTTP {status} {reason}: {error_body}")

        logger.debug(
            "TTS: %d байт, формат=%s", len(audio), self.tts_response_format
        )
        return audio

    def close(self) -> None:
        """Закрыть keep-alive соединение текущего потока."""
        self._drop_connection()

    @staticmethod
    def _build_multipart(
        fields: List[tuple],
        file_name: str,
        file_content_type: str,
        file_data: bytes,
        boundary: str,
    ) -> bytes:
        lines: List[bytes] = []
        crlf = "\r\n".encode()
        for key, value in fields:
            lines.append(f"--{boundary}".encode())
            lines.append(crlf)
            lines.append(
                f'Content-Disposition: form-data; name="{key}"'.encode()
            )
            lines.append(crlf)
            lines.append(crlf)
            lines.append(str(value).encode())
            lines.append(crlf)
        lines.append(f"--{boundary}".encode())
        lines.append(crlf)
        lines.append(
            (
                f'Content-Disposition: form-data; name="file"; '
                f'filename="{file_name}"'
            ).encode()
        )
        lines.append(crlf)
        lines.append(f"Content-Type: {file_content_type}".encode())
        lines.append(crlf)
        lines.append(crlf)
        lines.append(file_data)
        lines.append(crlf)
        lines.append(f"--{boundary}--".encode())
        lines.append(crlf)
        return b"".join(lines)


__all__ = ["AiClient", "AiClientError", "ChatResponse"]