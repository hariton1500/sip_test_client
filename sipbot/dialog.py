"""Голосовой диалоговый бот: VAD → ASR → LLM → TTS.

``DialogBot`` занимает место обычного бота в ``CallSession``. Входящие кадры
он отдаёт в ``EnergyVAD``; готовую реплику — в отдельный поток ASR/LLM/TTS.
Аудио ответа бот выдаёт **по одному кадру** на каждый входящий кадр, поэтому
не переполняет исходящую очередь и может мгновенно остановить проигрывание
при перебивании.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional

from .audio_utils import FRAME_BYTES, SILENCE_BYTE, iter_frames
from .transcript import BOT, CALLER, SYSTEM, TranscriptStore
from .tts import TtsService
from .vad import EnergyVAD

logger = logging.getLogger(__name__)

DEFAULT_GREETING = (
    "Здравствуйте! Вы позвонили в службу поддержки. Чем я могу вам помочь?"
)
DEFAULT_FALLBACK = "Извините, не удалось получить ответ. Повторите, пожалуйста."


def _env_int(
    source: Mapping[str, str], name: str, default: int, minimum: int = 0
) -> int:
    value = str(source.get(name, "") or "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Некорректное значение %s=%r, использую %d", name, value, default)
        return default
    return max(minimum, parsed)


def _env_float(
    source: Mapping[str, str], name: str, default: float, minimum: float = 0.0
) -> float:
    value = str(source.get(name, "") or "").strip()
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        logger.warning("Некорректное значение %s=%r, использую %.2f", name, value, default)
        return default
    return max(minimum, parsed)


def _env_bool(source: Mapping[str, str], name: str, default: bool) -> bool:
    value = str(source.get(name, "") or "").strip().lower()
    if not value:
        return default
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    logger.warning("Некорректное значение %s=%r, использую %s", name, value, default)
    return default


@dataclass(frozen=True)
class DialogConfig:
    """Настройки диалога, читаемые из переменных окружения."""

    greeting: str = DEFAULT_GREETING
    fallback_reply: str = DEFAULT_FALLBACK
    history_turns: int = 12
    max_reply_chars: int = 1200
    barge_in: bool = True
    vad_speech_ms: int = 200
    vad_silence_ms: int = 900
    vad_max_utterance_ms: int = 10000
    vad_rms_threshold: float = 0.02
    tts_cache_dir: Optional[Path] = Path("cache/tts")

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "DialogConfig":
        source: Mapping[str, str] = os.environ if environ is None else environ
        cache_dir = str(source.get("TTS_CACHE_DIR", "") or "").strip()
        return cls(
            greeting=str(source.get("AI_GREETING", DEFAULT_GREETING) or "").strip(),
            fallback_reply=str(
                source.get("AI_FALLBACK_REPLY", DEFAULT_FALLBACK) or ""
            ).strip(),
            history_turns=_env_int(source, "DIALOG_HISTORY_TURNS", 12, minimum=1),
            max_reply_chars=_env_int(
                source, "DIALOG_MAX_REPLY_CHARS", 1200, minimum=80
            ),
            barge_in=_env_bool(source, "DIALOG_BARGE_IN", True),
            vad_speech_ms=_env_int(source, "VAD_SPEECH_MS", 200, minimum=20),
            vad_silence_ms=_env_int(source, "VAD_SILENCE_MS", 900, minimum=20),
            vad_max_utterance_ms=_env_int(
                source, "VAD_MAX_UTTERANCE_MS", 10000, minimum=500
            ),
            vad_rms_threshold=_env_float(
                source, "VAD_RMS_THRESHOLD", 0.02, minimum=0.0
            ),
            tts_cache_dir=Path(cache_dir) if cache_dir else None,
        )

    def make_vad(self) -> EnergyVAD:
        return EnergyVAD(
            speech_ms=self.vad_speech_ms,
            silence_ms=self.vad_silence_ms,
            max_utterance_ms=self.vad_max_utterance_ms,
            rms_threshold=self.vad_rms_threshold,
        )


@dataclass(frozen=True)
class LatencyReport:
    """Задержки одного ответа в миллисекундах."""

    call_id: str
    text: str
    asr_ms: float
    llm_ms: float
    tts_ms: float
    total_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "text": self.text,
            "asr_ms": round(self.asr_ms, 1),
            "llm_ms": round(self.llm_ms, 1),
            "tts_ms": round(self.tts_ms, 1),
            "total_ms": round(self.total_ms, 1),
        }


@dataclass(frozen=True)
class _Utterance:
    wav_bytes: bytes
    speech_end: float
    epoch: int
    final: bool = False


class DialogBot:
    """ASR → LLM → TTS с потокобезопасной исходящей очередью."""

    def __init__(
        self,
        ai,
        system_prompt: str,
        store: Optional[TranscriptStore] = None,
        *,
        config: Optional[DialogConfig] = None,
        vad: Optional[EnergyVAD] = None,
        tts: Optional[TtsService] = None,
        greeting_audio: Optional[bytes] = None,
        language: Optional[str] = None,
        asr_prompt: Optional[str] = None,
    ) -> None:
        if ai is None:
            raise ValueError("DialogBot требует AiClient")
        if not (system_prompt or "").strip():
            raise ValueError("DialogBot требует непустой system prompt")

        self.ai = ai
        self.system_prompt = system_prompt.strip()
        self.store = store
        self.config = config or DialogConfig.from_env()
        self._vad = vad or self.config.make_vad()
        self._tts = tts or TtsService(ai, cache_dir=self.config.tts_cache_dir)
        self._language = language
        self._asr_prompt = asr_prompt

        self._call_id = ""
        self._closed = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._queue: "queue.Queue[_Utterance]" = queue.Queue()
        self._out: Deque[bytes] = deque()
        self._lock = threading.RLock()
        self._history: List[Dict[str, str]] = []
        self._epoch = 0
        self._turn_busy = False
        self._barge_in_triggered = False
        self._ended = False
        self._reports: List[LatencyReport] = []

        self._greeting = self.config.greeting
        self._greeting_audio = greeting_audio

    # ------------------------------------------------------------------
    # Bot API
    # ------------------------------------------------------------------
    def on_call_started(self, caller_id: str, call_id: str) -> Optional[bytes]:
        self._call_id = call_id
        self._vad.reset()
        with self._lock:
            self._history.clear()
            self._out.clear()
            self._epoch = 0
            self._turn_busy = False
            self._barge_in_triggered = False
            self._ended = False

        self._publish(SYSTEM, f"Входящий звонок: {caller_id or 'неизвестный'}")
        if self._worker is None:
            self._worker = threading.Thread(
                target=self._worker_loop, name="sipbot-dialog", daemon=True
            )
            self._worker.start()

        if self._greeting:
            if self._greeting_audio is None:
                try:
                    audio = self._tts.preload(self._greeting)
                except Exception:  # noqa: BLE001 - звонок не должен падать из-за TTS
                    logger.exception("Не удалось синтезировать приветствие")
                    audio = b""
            else:
                # Пустой bytes означает «синтез уже пробовали и он не удался»;
                # не повторяем сетевой вызов на каждом звонке.
                audio = self._greeting_audio
            if audio:
                self._enqueue_audio(audio)
                self._publish(BOT, self._greeting)
        # Приветствие отдаётся не через return, а по кадрам из on_frame:
        # так его можно прервать и не переполнять исходящую очередь.
        return None

    def on_frame(self, frame: bytes) -> Optional[bytes]:
        if self._closed.is_set():
            return None

        if frame:
            try:
                ready = self._vad.feed(frame)
            except Exception:  # noqa: BLE001 - VAD не должен рвать звонок
                logger.exception("Ошибка VAD в DialogBot")
                self._vad.reset()
                ready = False

            if self.config.barge_in and self._vad.speech_active:
                if not self._barge_in_triggered:
                    self._barge_in_triggered = True
                    self._interrupt_if_speaking()
            elif not self._vad.speech_active:
                self._barge_in_triggered = False

            if ready:
                try:
                    wav_bytes = self._vad.flush()
                    speech_end = time.monotonic() - self._vad.silence_ms / 1000.0
                    self._barge_in_triggered = False
                    self._enqueue_utterance(wav_bytes, speech_end)
                except Exception:  # noqa: BLE001
                    logger.exception("Ошибка завершения реплики")

        return self._pop_output()

    def on_call_ended(self) -> None:
        with self._lock:
            if self._ended:
                return
            self._ended = True

        if self._vad.has_speech:
            try:
                wav_bytes = self._vad.flush()
                self._enqueue_utterance(wav_bytes, time.monotonic(), final=True)
            except Exception:  # noqa: BLE001
                logger.exception("Не удалось сохранить последнюю реплику")
        else:
            self._vad.reset()

        self._publish(SYSTEM, "Звонок завершён")
        logger.info("Диалог завершён: %s", self._call_id)

    def close(self, grace: float = 2.0) -> None:
        """Остановить бота, дав ASR шанс дораспознать последнюю реплику."""
        self._closed.set()
        self._vad.close()
        self._queue.put(_Utterance(b"", 0.0, self._epoch, final=True))
        worker = self._worker
        if worker is not None and worker.is_alive() and grace > 0:
            worker.join(grace)
        with self._lock:
            self._out.clear()

    # ------------------------------------------------------------------
    # Очередь и перебивание
    # ------------------------------------------------------------------
    @property
    def call_id(self) -> str:
        return self._call_id

    @property
    def pending_output_frames(self) -> int:
        with self._lock:
            return len(self._out)

    @property
    def latency_reports(self) -> List[LatencyReport]:
        with self._lock:
            return list(self._reports)

    @property
    def last_latency(self) -> Optional[LatencyReport]:
        with self._lock:
            return self._reports[-1] if self._reports else None

    def _enqueue_utterance(
        self, wav_bytes: bytes, speech_end: float, final: bool = False
    ) -> None:
        if not wav_bytes:
            return
        with self._lock:
            epoch = self._epoch
        self._queue.put(_Utterance(wav_bytes, speech_end, epoch, final=final))

    def _enqueue_audio(self, audio_u8: bytes) -> None:
        if not audio_u8:
            return
        frames = list(iter_frames(audio_u8, FRAME_BYTES, pad_byte=SILENCE_BYTE))
        with self._lock:
            self._out.extend(frames)

    def _pop_output(self) -> Optional[bytes]:
        with self._lock:
            return self._out.popleft() if self._out else None

    def _interrupt_if_speaking(self) -> None:
        with self._lock:
            if not self._turn_busy and not self._out:
                return
            cleared = len(self._out)
            self._out.clear()
            self._epoch += 1
        logger.info(
            "Перебивание [%s]: остановлен ответ, отброшено кадров: %d",
            self._call_id,
            cleared,
        )
        self._publish(SYSTEM, "Абонент перебил бота — ответ остановлен")

    # ------------------------------------------------------------------
    # ASR / LLM / TTS
    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while True:
            try:
                utterance = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._closed.is_set():
                    return
                continue

            if not utterance.wav_bytes:
                if self._closed.is_set():
                    return
                continue

            with self._lock:
                self._turn_busy = True
            try:
                self._process(utterance)
            except Exception:  # noqa: BLE001 - один сбой не должен убивать диалог
                logger.exception("Ошибка обработки реплики [%s]", self._call_id)
                self._publish(SYSTEM, "Ошибка обработки реплики")
            finally:
                with self._lock:
                    self._turn_busy = False

    def _process(self, utterance: _Utterance) -> None:
        asr_started = time.monotonic()
        try:
            text = self.ai.transcribe(
                utterance.wav_bytes,
                language=self._language,
                prompt=self._asr_prompt,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ошибка ASR [%s]", self._call_id)
            self._publish(SYSTEM, f"Не удалось распознать реплику: {exc}")
            return
        asr_ms = (time.monotonic() - asr_started) * 1000.0

        text = " ".join((text or "").split())
        if not text:
            logger.info("ASR [%s]: пустая реплика", self._call_id)
            return

        self._publish(CALLER, text)
        logger.info("ASR [%s]: %s", self._call_id, text)
        with self._lock:
            self._history.append({"role": "user", "content": text})
            self._trim_history_locked()
            stale = utterance.epoch != self._epoch

        # После hangup успеваем распознать последнюю реплику, но уже не
        # тратим время на LLM/TTS.
        if self._closed.is_set() or stale:
            return

        llm_started = time.monotonic()
        try:
            response = self.ai.chat(self._messages())
            content = (getattr(response, "content", "") or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ошибка LLM [%s]", self._call_id)
            self._publish(SYSTEM, f"Ошибка LLM: {exc}")
            content = self.config.fallback_reply
        llm_ms = (time.monotonic() - llm_started) * 1000.0

        content = self._prepare_reply(content)
        if not content:
            content = self.config.fallback_reply
        if not content:
            return

        with self._lock:
            if self._closed.is_set() or utterance.epoch != self._epoch:
                reason = "звонок завершён" if self._closed.is_set() else "перебивание"
                logger.info("Ответ отменён (%s) [%s]", reason, self._call_id)
                return
            self._history.append({"role": "assistant", "content": content})
            self._trim_history_locked()

        self._publish(BOT, content)
        logger.info("LLM [%s]: %s", self._call_id, content)

        tts_started = time.monotonic()
        try:
            audio = self._tts.audio_u8(content)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ошибка TTS [%s]", self._call_id)
            self._publish(SYSTEM, f"Ошибка синтеза речи: {exc}")
            return
        tts_ms = (time.monotonic() - tts_started) * 1000.0

        with self._lock:
            if self._closed.is_set() or utterance.epoch != self._epoch:
                reason = "звонок завершён" if self._closed.is_set() else "перебивание"
                logger.info("TTS отменён (%s) [%s]", reason, self._call_id)
                return
            self._enqueue_audio(audio)

        total_ms = (time.monotonic() - utterance.speech_end) * 1000.0
        report = LatencyReport(
            call_id=self._call_id,
            text=content,
            asr_ms=asr_ms,
            llm_ms=llm_ms,
            tts_ms=tts_ms,
            total_ms=total_ms,
        )
        with self._lock:
            self._reports.append(report)
        logger.info(
            "Задержка ответа [%s]: ASR %.2f с, LLM %.2f с, TTS %.2f с, "
            "всего от конца речи %.2f с",
            self._call_id,
            asr_ms / 1000.0,
            llm_ms / 1000.0,
            tts_ms / 1000.0,
            total_ms / 1000.0,
        )

    def _messages(self) -> List[Dict[str, str]]:
        with self._lock:
            history = list(self._history)
        return [{"role": "system", "content": self.system_prompt}, *history]

    def _trim_history_locked(self) -> None:
        limit = max(2, self.config.history_turns * 2)
        if len(self._history) > limit:
            self._history = self._history[-limit:]

    def _prepare_reply(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        # LLM иногда отвечает Markdown-ом; в телефонном TTS разметка лишняя.
        text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"[*_#`>]+", "", text)
        text = " ".join(text.split())
        limit = self.config.max_reply_chars
        if len(text) <= limit:
            return text

        cut = text[:limit]
        sentence_end = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        if sentence_end >= limit // 2:
            return cut[: sentence_end + 1].strip()
        space = cut.rfind(" ")
        return (cut[:space] if space > 0 else cut).strip()

    def _publish(self, role: str, text: str) -> None:
        if self.store is None or not text:
            return
        try:
            self.store.publish(role, text, call_id=self._call_id)
        except Exception:  # noqa: BLE001 - UI не должен рвать звонок
            logger.exception("Не удалось опубликовать сообщение в UI")


__all__ = [
    "DEFAULT_FALLBACK",
    "DEFAULT_GREETING",
    "DialogBot",
    "DialogConfig",
    "LatencyReport",
]

