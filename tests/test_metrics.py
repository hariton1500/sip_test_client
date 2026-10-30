"""Тесты разбора, агрегации и записи метрик задержек."""

from __future__ import annotations

import json

from sipbot.metrics import (
    LatencySample,
    MetricsWriter,
    format_report,
    load_samples,
    parse_jsonl,
    parse_log_text,
    summarize,
)

LOG = (
    "12:00:01 INFO    sipbot.dialog: Задержка ответа [call-1]: очередь 0.01 с, "
    "ASR 0.62 с, LLM 1.35 с, TTS 0.88 с, первый звук 1.55 с, "
    "всего от конца речи 2.60 с\n"
    "12:00:01 INFO    sipbot.dialog: Детали задержек [call-1]: "
    "LLM раунды=[652.3, 701.5] мс, тулы=[95.4] мс, "
    "TTS предложения=[410.2, 233.9] мс\n"
)


def _sample(**overrides) -> LatencySample:
    values = dict(
        call_id="call-1",
        queue_ms=10.0,
        asr_ms=500.0,
        llm_ms=1000.0,
        tts_ms=500.0,
        total_ms=2010.0,
        first_audio_ms=900.0,
    )
    values.update(overrides)
    return LatencySample(**values)


# ----------------------------------------------------------------------
# Разбор лога
# ----------------------------------------------------------------------
def test_parse_log_line_with_details():
    samples = parse_log_text(LOG)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.call_id == "call-1"
    assert sample.queue_ms == 10.0
    assert sample.asr_ms == 620.0
    assert sample.llm_ms == 1350.0
    assert sample.tts_ms == 880.0
    assert sample.first_audio_ms == 1550.0
    assert sample.total_ms == 2600.0
    assert sample.llm_rounds_ms == (652.3, 701.5)
    assert sample.tool_ms == 95.4
    assert sample.tts_chunks_ms == (410.2, 233.9)
    assert sample.tts_first_chunk_ms == 410.2


def test_parse_log_accepts_comma_decimals():
    line = (
        "Задержка ответа [call-2]: очередь 0,00 с, ASR 0,62 с, LLM 1,35 с, "
        "TTS 0,88 с, первый звук 1,55 с, всего от конца речи 2,60 с"
    )

    sample = parse_log_text(line)[0]

    assert sample.call_id == "call-2"
    assert sample.asr_ms == 620.0
    assert sample.total_ms == 2600.0


def test_parse_log_links_details_before_main_line():
    details, main = LOG.splitlines()
    sample = parse_log_text(main + "\n" + details)[0]

    assert sample.llm_rounds_ms == (652.3, 701.5)
    assert sample.tts_chunks_ms == (410.2, 233.9)


def test_parse_log_ignores_garbage_and_keeps_allocation():
    text = "\n".join(
        [
            "Traceback (most recent call last):",
            "  File \"dialog.py\", line 1, in _process",
            "Задержка ответа [call-3]: очередь 0.02 с, ASR 0.10 с, LLM 0.20 с, "
            "TTS 0.30 с, первый звук 0.40 с, всего от конца речи 0.50 с",
            "Задержка ответа [call-3] без чисел",
            "мусор",
        ]
    )

    samples = parse_log_text(text)

    assert len(samples) == 1
    assert samples[0].call_id == "call-3"
    assert samples[0].llm_ms == 200.0


def test_parse_log_without_details_has_empty_breakdown():
    sample = parse_log_text(LOG.splitlines()[0])[0]

    assert sample.llm_rounds_ms == ()
    assert sample.tool_ms == 0.0
    assert sample.tts_chunks_ms == ()


# ----------------------------------------------------------------------
# JSONL и автоопределение формата
# ----------------------------------------------------------------------
def test_parse_jsonl_skips_broken_lines():
    payload = _sample().to_dict()
    text = "\n".join([json.dumps(payload), "не json", '{"foo": 1}'])

    samples = parse_jsonl(text)

    assert [item.call_id for item in samples] == ["call-1"]
    assert samples[0].asr_ms == 500.0


def test_metrics_writer_appends_jsonl(tmp_path):
    writer = MetricsWriter(tmp_path / "nested" / "metrics.jsonl")

    writer.write(_sample())
    writer.write_all([_sample(call_id="call-2")])

    lines = writer.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["call_id"] == "call-1"
    assert "ts" in payload
    assert payload["total_ms"] == 2010.0


def test_metrics_writer_swallows_bad_path(tmp_path):
    """Ошибка записи не должна срывать звонок."""
    writer = MetricsWriter(tmp_path)  # каталог вместо файла

    writer.write(_sample())


def test_load_samples_detects_format(tmp_path):
    log_path = tmp_path / "call.log"
    log_path.write_text(LOG, encoding="utf-8")
    jsonl_path = tmp_path / "metrics.jsonl"
    jsonl_path.write_text(
        json.dumps(_sample().to_dict(), ensure_ascii=False) + "\n", encoding="utf-8"
    )

    from_log = load_samples(log_path)
    from_jsonl = load_samples(jsonl_path)

    assert len(from_log) == 1 and from_log[0].call_id == "call-1"
    assert len(from_jsonl) == 1 and from_jsonl[0].total_ms == 2010.0


# ----------------------------------------------------------------------
# Агрегация и отчёт
# ----------------------------------------------------------------------
def _three_samples():
    return [
        _sample(llm_rounds_ms=(600.0, 400.0)),
        _sample(
            call_id="call-2",
            asr_ms=700.0,
            llm_ms=1400.0,
            tts_ms=700.0,
            total_ms=2810.0,
            first_audio_ms=1200.0,
            llm_rounds_ms=(900.0,),
            tts_chunks_ms=(700.0,),
        ),
        _sample(
            call_id="call-3",
            asr_ms=300.0,
            llm_ms=600.0,
            tts_ms=300.0,
            total_ms=1210.0,
            first_audio_ms=600.0,
            llm_rounds_ms=(300.0, 200.0),
            tts_chunks_ms=(150.0, 150.0),
        ),
    ]


def test_summarize_computes_stages_and_shares():
    summary = summarize(_three_samples())

    assert summary["count"] == 3
    assert summary["calls"] == 3
    asr = summary["stages"]["asr_ms"]
    assert asr["mean"] == 500.0
    assert asr["median"] == 500.0
    assert asr["share_pct"] == 24.9
    llm = summary["stages"]["llm_ms"]
    assert llm["mean"] == 1000.0
    assert llm["median"] == 1000.0
    assert llm["p90"] == 1320.0
    assert llm["max"] == 1400.0
    assert summary["extras"]["first_audio_ms"]["median"] == 900.0


def test_summarize_groups_llm_rounds_and_tts_chunks():
    summary = summarize(_three_samples())

    rounds = summary["llm_rounds"]
    assert [row["index"] for row in rounds] == [1, 2]
    assert rounds[0]["n"] == 3
    assert rounds[0]["median"] == 600.0
    assert rounds[1]["n"] == 2
    assert rounds[1]["median"] == 300.0

    chunks = summary["tts_chunks"]
    assert [row["index"] for row in chunks] == [1, 2]
    assert chunks[0]["n"] == 2
    assert chunks[0]["median"] == 425.0


def test_summarize_accepts_report_like_objects():
    class _Report:
        def __init__(self, sample):
            self._sample = sample

        def to_dict(self):
            return self._sample.to_dict()

    summary = summarize([_Report(_sample())])

    assert summary["count"] == 1
    assert summary["total"]["median"] == 2010.0


def test_summarize_empty_reports():
    summary = summarize([])

    assert summary["count"] == 0
    assert format_report(summary) == "Метрик задержек нет"


def test_format_report_table_and_details():
    text = format_report(summarize(_three_samples()), details=True)

    assert "Реплик: 3" in text
    assert "очередь" in text and "ASR" in text and "TTS" in text
    assert "LLM: тулы" in text
    assert "LLM по раундам (мс)" in text
    assert "раунд 2" in text
    assert "TTS по предложениям (мс)" in text


def test_format_report_compact_is_one_line():
    line = format_report(summarize(_three_samples()), compact=True)

    assert "\n" not in line
    assert "реплик=3" in line
    assert "LLM медиана 1.00 с (p90 1.32)" in line
