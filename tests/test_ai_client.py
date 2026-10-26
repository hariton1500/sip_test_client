"""Тесты параметров LLM-запроса без сети."""

from __future__ import annotations

from sipbot.ai_client import AiClient


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
