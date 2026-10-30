"""Тесты локального TTS kokoro-ru без загрузки реальной модели."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from sipbot.ru_text import normalize_for_tts
from sipbot.tts import wav_bytes_to_pcm
from sipbot.tts_kokoro import (
    KokoroTts,
    KokoroUnavailable,
    _OnnxKokoroEngine,
    split_text_for_tts,
)


class FakeEngine:
    """Подмена ONNX/G2P: пишет фонемы и отдаёт 100 мс тона."""

    sample_rate = 24000

    def __init__(self, samples: int = 2400) -> None:
        self.samples = samples
        self.texts = []
        self.phonemes = []

    def phonemize(self, text):
        self.texts.append(text)
        return "a" * max(1, min(len(text), 500)), set()

    def render(self, ipa):
        self.phonemes.append(ipa)
        return np.full(self.samples, 0.25, dtype=np.float32)


def make_tts(tmp_path, engine=None):
    return KokoroTts(
        model_dir=tmp_path,
        engine=engine or FakeEngine(),
        normalizer=normalize_for_tts,
    )


def test_synthesize_returns_wav_and_normalizes_text(tmp_path):
    engine = FakeEngine()
    tts = make_tts(tmp_path, engine)

    wav = tts.synthesize("Баланс 1500 руб.")

    assert wav[:4] == b"RIFF"
    pcm, rate = wav_bytes_to_pcm(wav)
    assert rate == 24000
    assert len(pcm) == engine.samples * 2
    assert engine.texts == ["Баланс одна тысяча пятьсот рублей"]


def test_empty_text_does_not_touch_engine(tmp_path):
    engine = FakeEngine()
    tts = make_tts(tmp_path, engine)

    assert tts.synthesize("   ") == b""
    assert engine.texts == []


def test_long_text_is_split_and_joined_with_pause(tmp_path):
    engine = FakeEngine()
    tts = make_tts(tmp_path, engine)
    text = " ".join(["Очень длинное предложение номер"] * 20) + "."

    wav = tts.synthesize(text)

    pcm, rate = wav_bytes_to_pcm(wav)
    assert len(engine.texts) > 1
    # Каждый кусок по 100 мс + пауза 60 мс между кусками.
    expected = engine.samples * len(engine.texts) + int(
        rate * 0.06
    ) * (len(engine.texts) - 1)
    assert len(pcm) == expected * 2


class FakeInput:
    def __init__(self, name):
        self.name = name


class FakeSession:
    def __init__(self):
        self.feeds = None

    def get_inputs(self):
        return [FakeInput("input_ids"), FakeInput("style"), FakeInput("speed")]

    def run(self, output_names, feeds):
        self.feeds = feeds
        return [np.full((1, 4800), 0.5, dtype=np.float32)]


def test_onnx_engine_builds_feeds_and_style_index():
    session = FakeSession()
    style = np.arange(510 * 256, dtype=np.float32).reshape(510, 256)
    engine = _OnnxKokoroEngine(
        session=session,
        vocab={"a": 1, " ": 16},
        style=style,
        g2p=lambda text: ("a a", set()),
        speed=0.9,
    )

    audio = engine.render("a a")

    assert audio.shape == (1, 4800)
    assert set(session.feeds) == {"input_ids", "style", "speed"}
    assert session.feeds["input_ids"].tolist() == [[0, 1, 16, 1, 0]]
    # style выбирается по длине ipa - 1.
    assert session.feeds["style"].shape == (1, 256)
    assert session.feeds["style"][0, 0] == style[2, 0]
    assert session.feeds["speed"][0] == pytest.approx(0.9, rel=1e-6)


def test_ru_g2p_loader_forces_utf8_reading(tmp_path):
    module_path = tmp_path / "ru_g2p.py"
    data_path = tmp_path / "data.json"
    data_path.write_text('{"phoneme": "ˈa"}', encoding="utf-8")
    module_path.write_text(
        "from pathlib import Path\n"
        "VALUE = Path(__file__).with_name('data.json').read_text()\n",
        encoding="utf-8",
    )

    module = KokoroTts._load_module(module_path, "_test_ru_g2p_utf8")
    try:
        assert module.VALUE == '{"phoneme": "ˈa"}'
    finally:
        sys.modules.pop("_test_ru_g2p_utf8", None)


def test_missing_dependency_is_reported_before_download(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    tts = KokoroTts(model_dir=tmp_path, auto_download=True)

    with pytest.raises(KokoroUnavailable, match="onnxruntime"):
        tts.load()


def test_missing_model_files_fail_with_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "onnxruntime", types.SimpleNamespace()
    )
    tts = KokoroTts(model_dir=tmp_path, auto_download=False)

    with pytest.raises(KokoroUnavailable, match="Нет файлов модели"):
        tts.load()


def test_split_text_keeps_words_and_limit():
    text = "Первое предложение. " + " ".join(["слово"] * 100) + "."

    chunks = split_text_for_tts(text, limit=80)

    assert len(chunks) > 1
    assert all(len(chunk) <= 80 for chunk in chunks)
    assert " ".join(chunks).replace("  ", " ") == " ".join(text.split())
    assert not any(chunk.startswith(" ") for chunk in chunks)
