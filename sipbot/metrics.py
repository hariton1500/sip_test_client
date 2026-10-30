"""Метрики задержек ASR → LLM → TTS: сбор, разбор логов и агрегация.

Данные живут в двух видах:

* строка лога ``Задержка ответа [...]`` — то, что видно в консоли;
* JSONL (``MetricsWriter``) — то, по чему считается статистика.

``parse_log_text`` понимает оба формата, поэтому отчёт можно снять как по
свежему ``metrics.jsonl``, так и по сохранённому текстовому логу.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

#: Этапы пайплайна: их сумма вместе с ``queue_ms`` даёт ``total_ms``.
STAGES: Tuple[str, ...] = ("queue_ms", "asr_ms", "llm_ms", "tts_ms")
#: Метрики, которые не складываются в total, но важны для оценки UX.
EXTRAS: Tuple[str, ...] = ("first_audio_ms", "tool_ms", "tts_first_chunk_ms")

_NUM = r"-?\d+(?:[.,]\d+)?"

#: Строка, которую пишет ``DialogBot`` после каждой реплики.
LATENCY_RE = re.compile(
    r"Задержка ответа \[(?P<call_id>[^\]]*)\]:\s*"
    rf"очередь (?P<queue>{_NUM}) с,\s*"
    rf"ASR (?P<asr>{_NUM}) с,\s*"
    rf"LLM (?P<llm>{_NUM}) с,\s*"
    rf"TTS (?P<tts>{_NUM}) с,\s*"
    rf"первый звук (?P<first>{_NUM}) с,\s*"
    rf"всего от конца речи (?P<total>{_NUM}) с"
)

#: Детализация по раундам LLM и предложениям TTS.
DETAILS_RE = re.compile(
    r"Детали задержек \[(?P<call_id>[^\]]*)\]:\s*"
    rf"LLM раунды=\[(?P<rounds>[^\]]*)\] мс,\s*"
    rf"тулы=\[(?P<tools>[^\]]*)\] мс,\s*"
    rf"TTS предложения=\[(?P<chunks>[^\]]*)\] мс"
)


def _num(value: str) -> float:
    """Разобрать число из лога: ``1,23`` и ``1.23`` равнозначны."""
    return float(value.replace(",", ".").replace(" ", ""))


def _series(value: str) -> Tuple[float, ...]:
    """Разобрать список чисел вида ``612.3, 95.4`` (пустой список допустим)."""
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return tuple(_num(part) for part in parts)


def _percentile(values: Sequence[float], q: float) -> float:
    """Перцентиль с линейной интерполяцией (работает и при одном значении)."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * fraction)


@dataclass(frozen=True)
class LatencySample:
    """Задержки одной реплики в миллисекундах (без привязки к DialogBot)."""

    call_id: str = ""
    text: str = ""
    queue_ms: float = 0.0
    asr_ms: float = 0.0
    llm_ms: float = 0.0
    tts_ms: float = 0.0
    total_ms: float = 0.0
    first_audio_ms: float = 0.0
    llm_rounds_ms: Tuple[float, ...] = ()
    tool_ms: float = 0.0
    tts_chunks_ms: Tuple[float, ...] = ()
    tts_first_chunk_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "text": self.text,
            "queue_ms": round(self.queue_ms, 1),
            "asr_ms": round(self.asr_ms, 1),
            "llm_ms": round(self.llm_ms, 1),
            "tts_ms": round(self.tts_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "first_audio_ms": round(self.first_audio_ms, 1),
            "llm_rounds_ms": [round(value, 1) for value in self.llm_rounds_ms],
            "tool_ms": round(self.tool_ms, 1),
            "tts_chunks_ms": [round(value, 1) for value in self.tts_chunks_ms],
            "tts_first_chunk_ms": round(self.tts_first_chunk_ms, 1),
        }


def as_sample(source: Any) -> LatencySample:
    """Привести ``LatencyReport``/dict/``LatencySample`` к ``LatencySample``."""
    if isinstance(source, LatencySample):
        return source
    if hasattr(source, "to_dict"):
        source = source.to_dict()  # type: ignore[assignment]
    if not isinstance(source, Mapping):
        raise TypeError(f"Ожидался отчёт о задержках, получено {type(source)!r}")

    def value(name: str) -> float:
        try:
            return float(source.get(name) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def series(name: str) -> Tuple[float, ...]:
        raw = source.get(name) or ()
        if isinstance(raw, str):
            return _series(raw)
        return tuple(float(item) for item in raw)

    return LatencySample(
        call_id=str(source.get("call_id") or ""),
        text=str(source.get("text") or ""),
        queue_ms=value("queue_ms"),
        asr_ms=value("asr_ms"),
        llm_ms=value("llm_ms"),
        tts_ms=value("tts_ms"),
        total_ms=value("total_ms"),
        first_audio_ms=value("first_audio_ms"),
        llm_rounds_ms=series("llm_rounds_ms"),
        tool_ms=value("tool_ms"),
        tts_chunks_ms=series("tts_chunks_ms"),
        tts_first_chunk_ms=value("tts_first_chunk_ms"),
    )


def _replace(sample: LatencySample, details: Mapping[str, Any]) -> LatencySample:
    """Дополнить отчёт деталями, не теряя уже посчитанные поля."""
    payload = sample.to_dict()
    payload.update(details)
    return as_sample(payload)


def parse_log_text(text: str) -> List[LatencySample]:
    """Разобрать текстовый лог: основные строки + строки «Детали задержек»."""
    samples: List[LatencySample] = []
    by_call: Dict[str, int] = {}
    pending: Dict[str, Dict[str, Any]] = {}

    for line in text.splitlines():
        match = DETAILS_RE.search(line)
        if match is not None:
            call_id = match.group("call_id")
            chunks = _series(match.group("chunks"))
            details = {
                "llm_rounds_ms": _series(match.group("rounds")),
                "tool_ms": sum(_series(match.group("tools"))),
                "tts_chunks_ms": chunks,
                "tts_first_chunk_ms": chunks[0] if chunks else 0.0,
            }
            index = by_call.get(call_id)
            if index is None:
                pending[call_id] = details
            else:
                samples[index] = _replace(samples[index], details)
            continue

        match = LATENCY_RE.search(line)
        if match is None:
            continue

        call_id = match.group("call_id")
        sample = LatencySample(
            call_id=call_id,
            queue_ms=_num(match.group("queue")) * 1000.0,
            asr_ms=_num(match.group("asr")) * 1000.0,
            llm_ms=_num(match.group("llm")) * 1000.0,
            tts_ms=_num(match.group("tts")) * 1000.0,
            total_ms=_num(match.group("total")) * 1000.0,
            first_audio_ms=_num(match.group("first")) * 1000.0,
        )
        details = pending.pop(call_id, None)
        if details:
            sample = _replace(sample, details)
        by_call[call_id] = len(samples)
        samples.append(sample)

    return samples


def parse_jsonl(text: str) -> List[LatencySample]:
    """Разобрать JSONL: по одному отчёту в строке, битые строки пропускаются."""
    samples: List[LatencySample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping) and "total_ms" in payload:
            samples.append(as_sample(payload))
    return samples


def load_samples(path: PathLike) -> List[LatencySample]:
    """Прочитать отчёты из файла, определив формат по содержимому."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return parse_jsonl(text)
    return parse_log_text(text)


def _stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 1),
        "median": round(_percentile(values, 0.5), 1),
        "p90": round(_percentile(values, 0.9), 1),
        "min": round(min(values), 1),
        "max": round(max(values), 1),
    }


def _positional_stats(series: Iterable[Sequence[float]]) -> List[Dict[str, float]]:
    """Статистика по позициям: раунд LLM №1, №2… и предложение TTS №1, №2…"""
    rows = [tuple(values) for values in series]
    depth = max((len(row) for row in rows), default=0)
    result: List[Dict[str, float]] = []
    for index in range(depth):
        values = [row[index] for row in rows if len(row) > index]
        stats = _stats(values)
        stats["index"] = index + 1  # type: ignore[assignment]
        result.append(stats)
    return result


def summarize(reports: Iterable[Any]) -> Dict[str, Any]:
    """Посчитать count/mean/median/p90/min/max по каждому этапу."""
    samples = [as_sample(item) for item in reports]
    summary: Dict[str, Any] = {
        "count": len(samples),
        "calls": len({sample.call_id for sample in samples if sample.call_id}),
        "stages": {},
        "extras": {},
        "total": {},
        "llm_rounds": [],
        "tts_chunks": [],
    }
    if not samples:
        return summary

    mean_total = sum(sample.total_ms for sample in samples) / len(samples)
    summary["total"] = _stats([sample.total_ms for sample in samples])
    for name in STAGES:
        stats = _stats([getattr(sample, name) for sample in samples])
        stats["share_pct"] = (  # type: ignore[assignment]
            round(stats["mean"] / mean_total * 100.0, 1) if mean_total else 0.0
        )
        summary["stages"][name] = stats

    for name in EXTRAS:
        summary["extras"][name] = _stats([getattr(sample, name) for sample in samples])

    summary["llm_rounds"] = _positional_stats(
        sample.llm_rounds_ms for sample in samples
    )
    summary["tts_chunks"] = _positional_stats(
        sample.tts_chunks_ms for sample in samples
    )
    return summary


def summarize_reports(reports: Iterable[Any]) -> Dict[str, Any]:
    """Синоним ``summarize`` для читаемости в ``main.run_call``."""
    return summarize(reports)


_STAGE_LABELS = {
    "queue_ms": "очередь",
    "asr_ms": "ASR",
    "llm_ms": "LLM",
    "tts_ms": "TTS",
}

_EXTRA_LABELS = {
    "first_audio_ms": "первый звук",
    "tool_ms": "LLM: тулы",
    "tts_first_chunk_ms": "TTS: 1-е предложение",
}


def _seconds(value: float) -> str:
    return f"{value / 1000.0:.2f}"


def format_report(
    summary: Mapping[str, Any],
    *,
    details: bool = False,
    compact: bool = False,
) -> str:
    """Собрать отчёт: подробная таблица или одна строка для лога."""
    count = int(summary.get("count") or 0)
    if not count:
        return "Метрик задержек нет"

    stages: Mapping[str, Mapping[str, float]] = summary.get("stages") or {}
    extras: Mapping[str, Mapping[str, float]] = summary.get("extras") or {}

    if compact:
        parts = [f"реплик={count}", f"звонков={summary.get('calls', 0)}"]
        for name in STAGES:
            stats = stages.get(name) or {}
            parts.append(
                f"{_STAGE_LABELS[name]} медиана {_seconds(stats.get('median', 0.0))} с"
                f" (p90 {_seconds(stats.get('p90', 0.0))})"
            )
        for name in ("first_audio_ms", "tts_first_chunk_ms"):
            stats = extras.get(name) or {}
            if not stats.get("n"):
                continue
            parts.append(
                f"{_EXTRA_LABELS[name]} медиана {_seconds(stats.get('median', 0.0))} с"
                f" (p90 {_seconds(stats.get('p90', 0.0))})"
            )
        total = summary.get("total") or {}
        if total.get("n"):
            parts.append(
                f"всего медиана {_seconds(total.get('median', 0.0))} с"
                f" (p90 {_seconds(total.get('p90', 0.0))})"
            )
        return ", ".join(parts)

    lines = [f"Реплик: {count}, звонков: {summary.get('calls', 0)}"]
    header = (
        f"{'этап':<22}{'n':>4}{'сред':>8}{'медиана':>9}{'p90':>8}{'макс':>8}{'доля':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    def row(
        label: str, stats: Mapping[str, float], share: Optional[float] = None
    ) -> str:
        share_text = f"{share:>7.1f}%" if share is not None else f"{'-':>8}"
        return (
            f"{label:<22}{int(stats.get('n', 0)):>4}"
            f"{_seconds(stats.get('mean', 0.0)):>8}"
            f"{_seconds(stats.get('median', 0.0)):>9}"
            f"{_seconds(stats.get('p90', 0.0)):>8}"
            f"{_seconds(stats.get('max', 0.0)):>8}"
            f"{share_text}"
        )

    for name in STAGES:
        stats = stages.get(name) or {}
        lines.append(row(_STAGE_LABELS[name], stats, stats.get("share_pct")))

    total = summary.get("total") or {}
    if total.get("n"):
        lines.append(row("итого (от конца речи)", total))

    lines.append("")
    for name in EXTRAS:
        stats = extras.get(name)
        if stats and stats.get("n"):
            lines.append(row(_EXTRA_LABELS[name], stats))

    if details:
        rounds = summary.get("llm_rounds") or []
        if rounds:
            lines.append("")
            lines.append("LLM по раундам (мс):")
            for stats in rounds:
                lines.append(
                    f"  раунд {int(stats['index'])}: n={int(stats['n'])}, "
                    f"медиана {stats['median']:.1f}, p90 {stats['p90']:.1f}, "
                    f"макс {stats['max']:.1f}"
                )
        chunks = summary.get("tts_chunks") or []
        if chunks:
            lines.append("")
            lines.append("TTS по предложениям (мс):")
            for stats in chunks:
                lines.append(
                    f"  предложение {int(stats['index'])}: n={int(stats['n'])}, "
                    f"медиана {stats['median']:.1f}, p90 {stats['p90']:.1f}, "
                    f"макс {stats['max']:.1f}"
                )
    return "\n".join(lines)


class MetricsWriter:
    """Дописывает отчёты в JSONL, не срывая звонок при ошибке записи."""

    def __init__(self, path: PathLike) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def write(self, report: Any) -> None:
        payload = as_sample(report).to_dict()
        payload["ts"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        line = json.dumps(payload, ensure_ascii=False)
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except OSError:
            logger.warning(
                "Не удалось записать метрику задержек в %s", self._path, exc_info=True
            )

    def write_all(self, reports: Iterable[Any]) -> None:
        for report in reports:
            self.write(report)


__all__ = [
    "DETAILS_RE",
    "EXTRAS",
    "LATENCY_RE",
    "LatencySample",
    "MetricsWriter",
    "STAGES",
    "as_sample",
    "format_report",
    "load_samples",
    "parse_jsonl",
    "parse_log_text",
    "summarize",
    "summarize_reports",
]
