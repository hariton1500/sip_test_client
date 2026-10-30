"""Тесты выбора TTS-провайдера и отката на облачный синтез."""

from __future__ import annotations

import pytest

from sipbot import tts_provider
from sipbot.tts import TtsService
from sipbot.tts_kokoro import KokoroUnavailable
from sipbot.tts_provider import FallbackTts, build_tts_provider, build_tts_service


class FakeAi:
    tts_model = "cloud-tts"
    tts_voice = "Zephyr"
    tts_sample_rate = 24000

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = []

    def synthesize(self, text: str) -> bytes:
        if self.fail:
            raise RuntimeError("cloud down")
        self.calls.append(text)
        return b"RIFFcloud"


class FakeLocal:
    tts_model = "kokoro-ru:model"
    tts_voice = "sveta"
    tts_sample_rate = 24000

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = []

    def synthesize(self, text: str) -> bytes:
        if self.fail:
            raise RuntimeError("local down")
        self.calls.append(text)
        return b"RIFFlocal"


def test_cloud_is_default_and_returns_ai():
    ai = FakeAi()
    assert build_tts_provider(ai, {}) is ai


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="не поддерживается"):
        build_tts_provider(FakeAi(), {"TTS_PROVIDER": "piper"})


def test_explicit_kokoro_unavailable_is_configuration_error(monkeypatch):
    def broken(cls, environ=None):
        raise KokoroUnavailable("нет onnxruntime")

    monkeypatch.setattr(tts_provider.KokoroTts, "from_env", classmethod(broken))

    with pytest.raises(ValueError, match="requirements-kokoro"):
        build_tts_provider(FakeAi(), {"TTS_PROVIDER": "kokoro"})


def test_auto_falls_back_to_cloud_when_local_unavailable(monkeypatch):
    def broken(cls, environ=None):
        raise KokoroUnavailable("нет модели")

    monkeypatch.setattr(tts_provider.KokoroTts, "from_env", classmethod(broken))
    ai = FakeAi()

    assert build_tts_provider(ai, {"TTS_PROVIDER": "auto"}) is ai


def test_auto_uses_fallback_wrapper(monkeypatch):
    local = FakeLocal(fail=True)
    monkeypatch.setattr(
        tts_provider.KokoroTts,
        "from_env",
        classmethod(lambda cls, environ=None: local),
    )
    ai = FakeAi()

    provider = build_tts_provider(ai, {"TTS_PROVIDER": "auto"})

    assert isinstance(provider, FallbackTts)
    assert provider.synthesize("привет") == b"RIFFcloud"
    assert ai.calls == ["привет"]


def test_fallback_uses_local_when_it_works():
    provider = FallbackTts(FakeLocal(), FakeAi(fail=True))

    assert provider.synthesize("привет") == b"RIFFlocal"
    assert provider.tts_model == "kokoro-ru:model"


def test_build_tts_service_wraps_provider(monkeypatch, tmp_path):
    local = FakeLocal()
    monkeypatch.setattr(
        tts_provider.KokoroTts,
        "from_env",
        classmethod(lambda cls, environ=None: local),
    )

    service = build_tts_service(
        FakeAi(), cache_dir=tmp_path / "tts", environ={"TTS_PROVIDER": "kokoro"}
    )

    assert isinstance(service, TtsService)
    assert service.ai is local
