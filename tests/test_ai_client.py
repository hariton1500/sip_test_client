"""Тесты параметров LLM-запроса без сети."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sipbot.ai_client import AiClient, AiClientError


def test_chat_sends_generation_limits_tools_and_usage(monkeypatch):
    client = AiClient(
        "https://ai.example/v1",
        "secret",
        llm_max_tokens=120,
        llm_temperature=0.2,
    )
    captured = {}

    def fake_request(method, endpoint, payload=None, data=None, headers=None):
        captured["method"] = method
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "message": {
                        "content": "Здравствуйте!",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "end_call", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 20,
                "total_tokens": 1020,
            },
        }

    monkeypatch.setattr(client, "_request", fake_request)
    tools = [{"type": "function", "function": {"name": "end_call"}}]

    response = client.chat(
        [{"role": "user", "content": "Здравствуйте"}], tools=tools
    )

    assert captured["method"] == "POST"
    assert captured["endpoint"] == "/chat/completions"
    payload = captured["payload"]
    assert payload["max_tokens"] == 120
    assert payload["temperature"] == 0.2
    assert payload["tools"] == tools
    assert payload["tool_choice"] == "auto"
    assert response.content == "Здравствуйте!"
    assert response.finish_reason == "tool_calls"
    assert response.usage["total_tokens"] == 1020
    assert response.has_tool_calls is True


def test_from_env_reads_llm_limits_and_clamps_values():
    client = AiClient.from_env(
        {
            "OPENAI_BASE_URL": "https://ai.example/v1",
            "OPENAI_API_KEY": "secret",
            "LLM_MAX_TOKENS": "250",
            "LLM_TEMPERATURE": "0.4",
        }
    )

    assert client is not None
    assert client.llm_max_tokens == 250
    assert client.llm_temperature == 0.4


def test_chat_sends_provider_fallbacks_and_reasoning(monkeypatch):
    client = AiClient(
        "https://ai.example/v1",
        "secret",
        llm_model="gemini-3.1-flash-lite",
        llm_provider={"preferred_max_latency": {"p90": 1.2}},
        llm_service_tier="priority",
        llm_fallback_models=("claude-haiku-5.5", "deepseek-v4-flash", ""),
        llm_reasoning_effort="low",
    )
    captured = {}

    def fake_request(method, endpoint, payload=None, data=None, headers=None):
        captured["payload"] = payload
        return {
            "model": "claude-haiku-5.5",
            "service_tier": "priority",
            "choices": [{"message": {"content": "Готово"}}],
        }

    monkeypatch.setattr(client, "_request", fake_request)

    response = client.chat([{"role": "user", "content": "Привет"}])

    payload = captured["payload"]
    assert payload["model"] == "gemini-3.1-flash-lite"
    assert payload["models"] == [
        "gemini-3.1-flash-lite",
        "claude-haiku-5.5",
        "deepseek-v4-flash",
    ]
    assert payload["provider"] == {"preferred_max_latency": {"p90": 1.2}}
    assert payload["service_tier"] == "priority"
    assert payload["reasoning_effort"] == "low"
    assert response.content == "Готово"


def test_fallback_models_are_capped_to_five_total():
    client = AiClient(
        "https://ai.example/v1",
        "secret",
        llm_fallback_models=["a", "b", "c", "d", "e", "f"],
    )

    assert len(client.llm_fallback_models) == 4
    assert client.llm_fallback_models == ["a", "b", "c", "d"]


def test_from_env_defaults_to_fast_models():
    client = AiClient.from_env(
        {"OPENAI_BASE_URL": "https://ai.example/v1", "OPENAI_API_KEY": "secret"}
    )

    assert client is not None
    assert client.asr_model == "whisper-large-v3-turbo"
    assert client.llm_model == "gemini-3.1-flash-lite"
    assert client.llm_max_tokens == 150
    assert client.llm_provider == {"preferred_max_latency": {"p90": 1.2}}
    assert client.llm_fallback_models == []
    assert client.llm_service_tier == ""
    assert client.llm_reasoning_effort == ""


def test_from_env_reads_provider_routing():
    client = AiClient.from_env(
        {
            "OPENAI_BASE_URL": "https://ai.example/v1",
            "OPENAI_API_KEY": "secret",
            "LLM_PROVIDER_MAX_LATENCY_P90": "0.9",
            "LLM_PROVIDER_MIN_THROUGHPUT": "80",
            "LLM_PROVIDER_SORT": "Latency",
            "LLM_PROVIDER_ONLY": "google-ai-studio, google",
            "LLM_PROVIDER_IGNORE": "azure;openai",
            "LLM_FALLBACK_MODELS": "claude-haiku-5.5, deepseek-v4-flash",
            "LLM_SERVICE_TIER": " Priority ",
            "LLM_REASONING_EFFORT": "LOW",
        }
    )

    assert client is not None
    assert client.llm_provider == {
        "preferred_max_latency": {"p90": 0.9},
        "preferred_min_throughput": 80.0,
        "sort": "latency",
        "only": ["google-ai-studio", "google"],
        "ignore": ["azure", "openai"],
    }
    assert client.llm_fallback_models == ["claude-haiku-5.5", "deepseek-v4-flash"]
    assert client.llm_service_tier == "priority"
    assert client.llm_reasoning_effort == "low"


def test_provider_routing_can_be_disabled():
    client = AiClient.from_env(
        {
            "OPENAI_BASE_URL": "https://ai.example/v1",
            "OPENAI_API_KEY": "secret",
            "LLM_PROVIDER_MAX_LATENCY_P90": "0",
        }
    )

    assert client is not None
    assert client.llm_provider is None


def test_synthesize_returns_audio_and_sends_tts_payload(monkeypatch):
    client = AiClient("https://ai.example/v1", "secret")
    captured = {}

    def fake_raw_request(method, endpoint, body, headers, timeout):
        captured["method"] = method
        captured["endpoint"] = endpoint
        captured["payload"] = json.loads(body.decode("utf-8"))
        captured["timeout"] = timeout
        return 200, b"RIFF....", "OK"

    monkeypatch.setattr(client, "_raw_request", fake_raw_request)

    audio = client.synthesize("Здравствуйте")

    assert audio == b"RIFF...."
    assert captured["method"] == "POST"
    assert captured["endpoint"] == "/audio/speech"
    assert captured["payload"]["input"] == "Здравствуйте"
    assert captured["payload"]["model"] == client.tts_model


def test_request_error_contains_http_status(monkeypatch):
    client = AiClient("https://ai.example/v1", "secret")
    monkeypatch.setattr(
        client,
        "_raw_request",
        lambda *args, **kwargs: (503, b'{"error": "down"}', "Service Unavailable"),
    )

    with pytest.raises(AiClientError, match="HTTP 503 Service Unavailable"):
        client.chat([{"role": "user", "content": "Привет"}])


def test_requests_reuse_keep_alive_connection():
    connections = set()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802 - имя задаёт http.server
            connections.add(self.connection)
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            body = json.dumps(
                {"choices": [{"message": {"content": "ok"}}]}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = AiClient(
            f"http://127.0.0.1:{server.server_port}/v1", "secret"
        )
        assert client.chat([{"role": "user", "content": "раз"}]).content == "ok"
        assert client.chat([{"role": "user", "content": "два"}]).content == "ok"
        assert len(connections) == 1
        client.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
