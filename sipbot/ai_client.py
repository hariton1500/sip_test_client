"""OpenAI-совместимый клиент для ASR/LLM/TTS.

Работает через стандартную библиотеку urllib, чтобы не зависеть от requests/httpx.
Поддерживает любые OpenAI-совместимые API, в том числе AiTunnel.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class AiClientError(Exception):
    """Ошибка при обращении к AI API."""


@dataclass(frozen=True)
class ChatResponse:
    """Ответ LLM: текст и/или tool_calls."""

    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class AiClient:
    """Клиент ASR/LLM/TTS через OpenAI-совместимое API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        asr_model: str = "whisper-large-v3",
        llm_model: str = "deepseek-v4-flash",
        tts_model: str = "gemini-3.8-flash-lite-tts",
        tts_voice: str = "Zephyr",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.asr_model = asr_model
        self.llm_model = llm_model
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.timeout = timeout

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

        request = urllib.request.Request(
            url,
            data=body,
            headers=req_headers,
            method=method,
        )
        logger.debug("AI API %s %s", method, endpoint)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")[:500]
            raise AiClientError(
                f"HTTP {exc.code} {exc.reason}: {error_body}"
            ) from exc
        except Exception as exc:
            raise AiClientError(f"Ошибка запроса к {url}: {exc}") from exc

        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise AiClientError(f"екорректный JSON от {endpoint}: {exc}") from exc

    def transcribe(self, wav_bytes: bytes) -> str:
        """Распознать речь из WAV s16le. Возвращает текст."""
        boundary = f"----FormBoundary{uuid.uuid4().hex}"
        fields: List[tuple] = [("model", self.asr_model)]
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
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        result = self._request("POST", "/chat/completions", payload=payload)
        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls")
        logger.debug("LLM: content=%r tool_calls=%s", content, bool(tool_calls))
        return ChatResponse(content=content, tool_calls=tool_calls)

    def synthesize(self, text: str) -> bytes:
        """Синтезировать речь. Возвращает WAV PCM 16-bit mono."""
        payload = {
            "model": self.tts_model,
            "voice": self.tts_voice,
            "input": text,
            "response_format": "wav",
        }
        url = f"{self.base_url}/audio/speech"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "audio/wav,audio/*",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                wav_bytes = response.read()
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")[:500]
            raise AiClientError(
                f"HTTP {exc.code} {exc.reason}: {error_body}"
            ) from exc
        except Exception as exc:
            raise AiClientError(f"Ошибка TTS: {exc}") from exc

        logger.debug("TTS: %d байт", len(wav_bytes))
        return wav_bytes

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