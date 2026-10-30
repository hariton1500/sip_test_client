"""Голосовой диалоговый бот: VAD → ASR → LLM → TTS.

``DialogBot`` занимает место обычного бота в ``CallSession``. Входящие кадры
он отдаёт в ``EnergyVAD``; готовую реплику — в отдельный поток ASR/LLM/TTS.
Аудио ответа бот выдаёт **по одному кадру** на каждый входящий кадр, поэтому
не переполняет исходящую очередь и может мгновенно остановить проигрывание
при перебивании.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

from .audio_utils import FRAME_BYTES, SILENCE_BYTE, iter_frames
from .billing import BillingClient, BillingError
from .transcript import BOT, CALLER, SYSTEM, TranscriptStore
from .tts import TtsService
from .vad import EnergyVAD

logger = logging.getLogger(__name__)

DEFAULT_GREETING = (
    "Здравствуйте! Вы позвонили в службу поддержки. Чем я могу вам помочь?"
)
DEFAULT_FALLBACK = "Извините, не удалось получить ответ. Повторите, пожалуйста."

# Whisper любит «дописывать» финальные титры, если во входном WAV почти нет
# речи. Фильтруем только фразы, состоящие целиком из таких шаблонов, чтобы не
# потерять нормальные короткие ответы вроде «Спасибо.» или «Да.».
ASR_HALLUCINATION_PHRASES = (
    "продолжение следует",
    "субтитры сделал",
    "субтитры сделал dimatorzok",
    "субтитры создавал",
    "субтитры создавал dimatorzok",
    "редактор субтитров",
    "корректор",
    "спасибо за просмотр",
    "спасибо за внимание",
    "подписывайтесь на канал",
    "ставьте лайки",
)


def _normalize_for_match(text: str) -> str:
    normalized = text.lower().replace("ё", "е")
    normalized = re.sub(r"[^\w\s]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _is_asr_hallucination(text: str) -> bool:
    normalized = _normalize_for_match(text)
    if not normalized:
        return False
    for phrase in ASR_HALLUCINATION_PHRASES:
        phrase_normalized = _normalize_for_match(phrase)
        if not phrase_normalized:
            continue
        # Фраза целиком или та же фраза, повторённая несколько раз.
        if normalized == phrase_normalized:
            return True
        if not normalized.replace(phrase_normalized, "").strip():
            return True
    return False


BILLING_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_billing_info",
        "description": (
            "Получить баланс, тариф, остаток дней, статус сети и ремонта "
            "по номеру лицевого счёта абонента."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "string",
                    "description": "Номер лицевого счёта, только цифры",
                }
            },
            "required": ["account_id"],
        },
    },
}

END_CALL_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "end_call",
        "description": (
            "Завершить звонок после того, как абонент попрощался или разговор "
            "закончен. Перед вызовом обязательно попрощайтесь."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

DEFAULT_END_CALL_REPLY = "Всего доброго! До свидания."
DEFAULT_NO_INPUT_PROMPT = (
    "Вы ещё здесь? Если вопрос остался, скажите, пожалуйста."
)


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
    max_reply_sentences: int = 2
    max_tool_rounds: int = 3
    barge_in: bool = True
    vad_speech_ms: int = 200
    # 600 мс — компромисс для телефонии: заметно быстрее ответ, но ещё не
    # обрываем паузу внутри фразы. Было 900 мс.
    vad_silence_ms: int = 600
    vad_max_utterance_ms: int = 10000
    vad_rms_threshold: float = 0.02
    vad_pre_roll_ms: int = 200
    # 0 отключает соответствующее поведение.
    no_input_prompt_ms: int = 10000
    no_input_hangup_ms: int = 60000
    no_input_prompt: str = DEFAULT_NO_INPUT_PROMPT
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
            max_reply_sentences=_env_int(
                source, "DIALOG_MAX_REPLY_SENTENCES", 2, minimum=1
            ),
            max_tool_rounds=_env_int(
                source, "DIALOG_MAX_TOOL_ROUNDS", 3, minimum=1
            ),
            barge_in=_env_bool(source, "DIALOG_BARGE_IN", True),
            vad_speech_ms=_env_int(source, "VAD_SPEECH_MS", 200, minimum=20),
            vad_silence_ms=_env_int(source, "VAD_SILENCE_MS", 600, minimum=20),
            vad_max_utterance_ms=_env_int(
                source, "VAD_MAX_UTTERANCE_MS", 10000, minimum=500
            ),
            vad_rms_threshold=_env_float(
                source, "VAD_RMS_THRESHOLD", 0.02, minimum=0.0
            ),
            vad_pre_roll_ms=_env_int(
                source, "VAD_PREROLL_MS", 200, minimum=0
            ),
            no_input_prompt_ms=_env_int(
                source, "DIALOG_NO_INPUT_PROMPT_MS", 10000, minimum=0
            ),
            no_input_hangup_ms=_env_int(
                source, "DIALOG_NO_INPUT_HANGUP_MS", 60000, minimum=0
            ),
            no_input_prompt=str(
                source.get("DIALOG_NO_INPUT_PROMPT", DEFAULT_NO_INPUT_PROMPT)
                or ""
            ).strip(),
            tts_cache_dir=Path(cache_dir) if cache_dir else None,
        )

    def make_vad(self) -> EnergyVAD:
        return EnergyVAD(
            speech_ms=self.vad_speech_ms,
            silence_ms=self.vad_silence_ms,
            max_utterance_ms=self.vad_max_utterance_ms,
            rms_threshold=self.vad_rms_threshold,
            pre_roll_ms=self.vad_pre_roll_ms,
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
    # Ожидание в очереди до старта ASR: без него сумма этапов не сходится
    # с total_ms при параллельных/запоздавших репликах.
    queue_ms: float = 0.0
    # Время от конца речи до первого готового аудио. При синтезе по
    # предложениям меньше total_ms, потому что second sentence ещё считается.
    first_audio_ms: float = 0.0
    # Детализация: сколько занял каждый раунд LLM (tool-цикл) и каждое
    # предложение TTS. llm_ms — их сумма, tts_ms — сумма чанков.
    llm_rounds_ms: Tuple[float, ...] = ()
    tool_ms: float = 0.0
    tts_chunks_ms: Tuple[float, ...] = ()
    tts_first_chunk_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "text": self.text,
            "asr_ms": round(self.asr_ms, 1),
            "llm_ms": round(self.llm_ms, 1),
            "tts_ms": round(self.tts_ms, 1),
            "queue_ms": round(self.queue_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "first_audio_ms": round(self.first_audio_ms, 1),
            "llm_rounds_ms": [round(value, 1) for value in self.llm_rounds_ms],
            "tool_ms": round(self.tool_ms, 1),
            "tts_chunks_ms": [round(value, 1) for value in self.tts_chunks_ms],
            "tts_first_chunk_ms": round(self.tts_first_chunk_ms, 1),
        }


@dataclass(frozen=True)
class _Utterance:
    wav_bytes: bytes
    speech_end: float
    epoch: int
    final: bool = False
    queued_at: float = 0.0


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
        billing: Optional[BillingClient] = None,
        allow_end_call: bool = True,
        end_call_reply: str = DEFAULT_END_CALL_REPLY,
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
        self.billing = billing
        self._end_call_reply = end_call_reply
        self._tools: List[Dict[str, Any]] = [BILLING_TOOL]
        if allow_end_call:
            self._tools.append(END_CALL_TOOL)

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
        # Абонент начал говорить, пока ответ ещё готовится (аудио не играет).
        self._pending_speech = False
        self._end_call_requested = False
        self._ended = False
        self._reports: List[LatencyReport] = []
        # No-input: отчёт тишины начинается после того, как бот договорил.
        self._quiet_since = 0.0
        self._bot_speaking = False
        self._idle_prompt_sent = False
        self._idle_thread: Optional[threading.Thread] = None

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
            self._pending_speech = False
            self._end_call_requested = False
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

        with self._lock:
            self._quiet_since = time.monotonic()
            self._bot_speaking = bool(self._out)
            self._idle_prompt_sent = False
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

            if self._vad.speech_active:
                with self._lock:
                    self._quiet_since = time.monotonic()
                    self._idle_prompt_sent = False
                if self.config.barge_in and not self._barge_in_triggered:
                    self._barge_in_triggered = True
                    self._interrupt_if_speaking()
            else:
                self._barge_in_triggered = False

            if ready:
                try:
                    wav_bytes = self._vad.flush()
                    info = self._vad.last_flush_info
                    trailing_ms = (
                        info.trailing_silence_ms
                        if info is not None
                        else self._vad.silence_ms
                    )
                    # perf_counter, чтобы метрики и очередь были в одной шкале.
                    speech_end = time.perf_counter() - trailing_ms / 1000.0
                    if info is not None:
                        logger.info(
                            "VAD [%s]: речь %.2f с, WAV %.2f с, хвост %.2f с "
                            "(%s)",
                            self._call_id,
                            info.speech_ms / 1000.0,
                            info.buffered_ms / 1000.0,
                            info.trailing_silence_ms / 1000.0,
                            info.reason,
                        )
                    self._barge_in_triggered = False
                    self._enqueue_utterance(wav_bytes, speech_end)
                except Exception:  # noqa: BLE001
                    logger.exception("Ошибка завершения реплики")

        self._check_no_input(time.monotonic())
        return self._pop_output()

    def _check_no_input(self, now: float) -> None:
        """Подсказать абоненту при тишине и затем завершить звонок."""
        prompt_ms = self.config.no_input_prompt_ms
        hangup_ms = self.config.no_input_hangup_ms
        prompt = self.config.no_input_prompt
        start_prompt = False
        hangup = False
        with self._lock:
            if self._closed.is_set():
                return
            if self._out:
                self._bot_speaking = True
            elif self._bot_speaking:
                # Бот только что договорил — с этого момента считаем тишину.
                self._bot_speaking = False
                self._quiet_since = now

            if self._ended or self._end_call_requested:
                return
            if (
                self._turn_busy
                or self._pending_speech
                or self._out
                or self._vad.speech_active
            ):
                return

            idle_ms = (now - self._quiet_since) * 1000.0
            if (
                prompt_ms > 0
                and prompt
                and not self._idle_prompt_sent
                and idle_ms >= prompt_ms
            ):
                self._idle_prompt_sent = True
                start_prompt = True
            elif (
                self._idle_prompt_sent
                and hangup_ms > 0
                and idle_ms >= hangup_ms
            ):
                self._end_call_requested = True
                hangup = True

        if start_prompt:
            self._idle_thread = threading.Thread(
                target=self._idle_prompt_worker,
                args=(self._call_id, now),
                name="sipbot-no-input",
                daemon=True,
            )
            self._idle_thread.start()
        if hangup:
            logger.info(
                "Нет ответа %.1f с — завершаю звонок [%s]",
                hangup_ms / 1000.0,
                self._call_id,
            )
            self._publish(SYSTEM, "Абонент не отвечает — завершаю звонок")

    def _idle_prompt_worker(self, call_id: str, started_at: float) -> None:
        prompt = self.config.no_input_prompt
        try:
            audio = self._tts.audio_u8(prompt)
        except Exception:  # noqa: BLE001 - тишина не должна рвать звонок
            logger.exception("Не удалось синтезировать no-input подсказку")
            return

        with self._lock:
            if (
                self._closed.is_set()
                or self._ended
                or self._call_id != call_id
                or self._pending_speech
                or self._turn_busy
                or self._out
                or self._quiet_since > started_at
            ):
                return
            self._enqueue_audio(audio)
            self._history.append({"role": "assistant", "content": prompt})
            self._trim_history_locked()

        self._publish(BOT, prompt)
        logger.info("No-input подсказка [%s]: %s", call_id, prompt)

    def on_call_ended(self) -> None:
        with self._lock:
            if self._ended:
                return
            self._ended = True

        if self._vad.has_speech:
            try:
                wav_bytes = self._vad.flush()
                info = self._vad.last_flush_info
                trailing_ms = (
                    info.trailing_silence_ms
                    if info is not None
                    else self._vad.silence_ms
                )
                speech_end = time.perf_counter() - trailing_ms / 1000.0
                if info is not None:
                    logger.info(
                        "VAD [%s]: финальная речь %.2f с, WAV %.2f с (%s)",
                        self._call_id,
                        info.speech_ms / 1000.0,
                        info.buffered_ms / 1000.0,
                        info.reason,
                    )
                self._enqueue_utterance(wav_bytes, speech_end, final=True)
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
    def end_call_requested(self) -> bool:
        """LLM попросил завершить звонок, и весь ответ уже отдан в мост."""
        with self._lock:
            return (
                self._end_call_requested
                and not self._turn_busy
                and not self._pending_speech
                and not self._out
            )

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
        self._queue.put(
            _Utterance(
                wav_bytes,
                speech_end,
                epoch,
                final=final,
                # perf_counter, а не monotonic: на Windows monotonic квантуется
                # миллисекундой, и ожидание в очереди измерялось как 0.0 мс.
                queued_at=time.perf_counter(),
            )
        )

    def _enqueue_audio(self, audio_u8: bytes) -> None:
        if not audio_u8:
            return
        frames = list(iter_frames(audio_u8, FRAME_BYTES, pad_byte=SILENCE_BYTE))
        with self._lock:
            self._out.extend(frames)

    def _pop_output(self) -> Optional[bytes]:
        with self._lock:
            return self._out.popleft() if self._out else None

    def _discard_queued_utterances(self) -> int:
        """Убрать из очереди реплики, которые стали неактуальны после barge-in."""
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return dropped
            dropped += 1

    def _interrupt_if_speaking(self) -> None:
        with self._lock:
            if self._out:
                # Реальный barge-in: звук уже играет, его нужно остановить.
                cleared = len(self._out)
                self._out.clear()
                self._epoch += 1
                self._pending_speech = False
                self._end_call_requested = False
                turn_busy = self._turn_busy
            elif self._turn_busy:
                # Бот ещё думает/синтезирует и ничего не сказал. Не отменяем
                # текущую реплику сразу: ASR-текст ещё пригодится истории.
                # Помечаем, что пришла новая речь, и не отдаём устаревший
                # ответ в LLM/TTS, когда он будет готов.
                if self._pending_speech:
                    return
                self._pending_speech = True
                # Прощание отменено: абонент ещё говорит, трубку не вешаем.
                self._end_call_requested = False
                cleared = 0
                turn_busy = True
            else:
                return

        if cleared:
            dropped = self._discard_queued_utterances()
            logger.info(
                "Перебивание [%s]: остановлен ответ, отброшено кадров: %d, "
                "реплик из очереди: %d",
                self._call_id,
                cleared,
                dropped,
            )
            self._publish(SYSTEM, "Абонент перебил бота — ответ остановлен")
        elif turn_busy:
            logger.info(
                "Перебивание [%s]: пришла новая речь до начала ответа, "
                "устаревший ответ будет отменён",
                self._call_id,
            )
            self._publish(
                SYSTEM, "Абонент продолжил говорить — ответ будет уточнён"
            )

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
                # Новая речь уже превратилась в эту реплику — флаг «пришла
                # новая речь» больше не должен отменять её собственный ответ.
                self._pending_speech = False
                self._turn_busy = True
            try:
                self._process(utterance)
            except Exception:  # noqa: BLE001 - один сбой не должен убивать диалог
                logger.exception("Ошибка обработки реплики [%s]", self._call_id)
                self._publish(SYSTEM, "Ошибка обработки реплики")
            finally:
                with self._lock:
                    self._turn_busy = False

    def _superseded_reason_locked(self, utterance: _Utterance) -> Optional[str]:
        """Причина отмены ответа; вызывается под ``self._lock``."""
        if self._closed.is_set():
            return "звонок завершён"
        if utterance.epoch != self._epoch:
            return "перебивание"
        if self._pending_speech:
            return "абонент продолжил говорить"
        return None

    def _abort_if_superseded(self, utterance: _Utterance, stage: str) -> bool:
        """Проверить, не устарел ли ответ до/во время ASR/LLM/TTS."""
        with self._lock:
            reason = self._superseded_reason_locked(utterance)
        if reason is None:
            return False
        with self._lock:
            # Отменённый ответ не должен тянуть за собой end_call.
            self._end_call_requested = False
        logger.info("%s отменён (%s) [%s]", stage, reason, self._call_id)
        return True

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Any:
        function = tool_call.get("function") or {}
        name = str(function.get("name") or "")
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = (
                raw_arguments
                if isinstance(raw_arguments, dict)
                else json.loads(raw_arguments)
            )
        except (TypeError, json.JSONDecodeError) as exc:
            logger.warning("Некорректные аргументы инструмента %s: %s", name, exc)
            return {"error": f"Некорректные аргументы: {exc}"}
        if not isinstance(arguments, dict):
            return {"error": "Аргументы инструмента должны быть объектом"}

        if name == "get_billing_info":
            account_id = str(
                arguments.get("account_id") or arguments.get("userId") or ""
            ).strip()
            if self.billing is None:
                logger.warning("BILLING: инструмент вызван без настроенного клиента")
                return {"error": "Биллинг не настроен"}
            try:
                data = self.billing.get_account(account_id)
            except BillingError as exc:
                logger.warning("BILLING: ошибка для счёта %r: %s", account_id, exc)
                return {"error": str(exc)}
            logger.info("BILLING: данные по счёту %s получены", account_id)
            return data

        if name == "end_call":
            with self._lock:
                self._end_call_requested = True
            logger.info("LLM попросил завершить звонок [%s]", self._call_id)
            return {"ok": True}

        logger.warning("LLM вызвал неизвестный инструмент: %r", name)
        return {"error": f"Неизвестный инструмент: {name}"}

    @staticmethod
    def _store_llm_stats(
        stats: Optional[Dict[str, Any]],
        rounds_ms: List[float],
        tools_ms: List[float],
    ) -> None:
        if stats is None:
            return
        stats["rounds_ms"] = list(rounds_ms)
        stats["tools_ms"] = list(tools_ms)
        stats["tool_ms"] = sum(tools_ms)

    def _chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        stats: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Выполнить tool-calls LLM и вернуть финальный ответ.

        Если передан ``stats``, заполняет ``rounds_ms`` (время каждого
        chat-запроса) и ``tools_ms`` (время каждого инструмента), чтобы
        в отчёте отделить сеть LLM от биллинга.
        """
        rounds_ms: List[float] = []
        tools_ms: List[float] = []
        for round_index in range(self.config.max_tool_rounds):
            chat_started = time.perf_counter()
            response = self.ai.chat(messages, tools=self._tools)
            rounds_ms.append((time.perf_counter() - chat_started) * 1000.0)
            if not response.has_tool_calls:
                self._store_llm_stats(stats, rounds_ms, tools_ms)
                return response

            tool_calls = response.tool_calls or []
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or None,
                    "tool_calls": tool_calls,
                }
            )
            for index, tool_call in enumerate(tool_calls):
                function = tool_call.get("function") or {}
                name = str(function.get("name") or "")
                tool_started = time.perf_counter()
                result = self._execute_tool_call(tool_call)
                tools_ms.append((time.perf_counter() - tool_started) * 1000.0)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id")
                        or f"call-{round_index}-{index}",
                        "name": name,
                        "content": json.dumps(
                            result, ensure_ascii=False, default=str
                        ),
                    }
                )
            logger.info(
                "LLM tool round %d [%s]: %s",
                round_index + 1,
                self._call_id,
                ", ".join(
                    str((call.get("function") or {}).get("name") or "?")
                    for call in tool_calls
                ),
            )

        logger.warning(
            "LLM не завершил tool-цикл за %d раундов [%s]",
            self.config.max_tool_rounds,
            self._call_id,
        )
        chat_started = time.perf_counter()
        response = self.ai.chat(messages)
        rounds_ms.append((time.perf_counter() - chat_started) * 1000.0)
        self._store_llm_stats(stats, rounds_ms, tools_ms)
        return response

    def _process(self, utterance: _Utterance) -> None:
        # Та же шкала, что и ``queued_at``: иначе очередь считается по нулю.
        started_at = time.perf_counter()
        queue_ms = (
            (started_at - utterance.queued_at) * 1000.0
            if utterance.queued_at
            else 0.0
        )
        asr_started = started_at
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
        asr_ms = (time.perf_counter() - asr_started) * 1000.0

        text = " ".join((text or "").split())
        if not text:
            logger.info("ASR [%s]: пустая реплика", self._call_id)
            return
        if _is_asr_hallucination(text):
            logger.info(
                "ASR [%s]: отфильтрована галлюцинация Whisper: %r",
                self._call_id,
                text,
            )
            return

        self._publish(CALLER, text)
        logger.info("ASR [%s]: %s", self._call_id, text)
        with self._lock:
            self._history.append({"role": "user", "content": text})
            self._trim_history_locked()

        # После hangup успеваем распознать последнюю реплику, но уже не
        # тратим время на LLM/TTS. Если абонент продолжил говорить, его текст
        # остаётся в истории, а устаревший ответ не генерируем.
        if self._abort_if_superseded(utterance, "LLM"):
            return

        llm_started = time.perf_counter()
        llm_stats: Dict[str, Any] = {}
        try:
            response = self._chat_with_tools(self._messages(), stats=llm_stats)
            content = (getattr(response, "content", "") or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ошибка LLM [%s]", self._call_id)
            self._publish(SYSTEM, f"Ошибка LLM: {exc}")
            content = self.config.fallback_reply
        llm_ms = (time.perf_counter() - llm_started) * 1000.0

        content = self._prepare_reply(content)
        if not content and self._end_call_requested:
            # Модель вызвала end_call, но не написала прощание.
            content = self._end_call_reply
        if not content:
            content = self.config.fallback_reply
        if not content:
            return

        with self._lock:
            reason = self._superseded_reason_locked(utterance)
            if reason is not None:
                logger.info("Ответ отменён (%s) [%s]", reason, self._call_id)
                return
            self._history.append({"role": "assistant", "content": content})
            self._trim_history_locked()

        self._publish(BOT, content)
        logger.info("LLM [%s]: %s", self._call_id, content)

        # Синтезируем по предложениям и отдаём первое сразу: для телефонии
        # важен time-to-first-audio, а не общая длительность синтеза.
        tts_started = time.perf_counter()
        first_audio_ms = 0.0
        tts_chunks_ms: List[float] = []
        for chunk in self._reply_chunks(content):
            chunk_started = time.perf_counter()
            try:
                audio = self._tts.audio_u8(chunk)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Ошибка TTS [%s]", self._call_id)
                self._publish(SYSTEM, f"Ошибка синтеза речи: {exc}")
                return
            tts_chunks_ms.append((time.perf_counter() - chunk_started) * 1000.0)
            with self._lock:
                reason = self._superseded_reason_locked(utterance)
                if reason is not None:
                    logger.info("TTS отменён (%s) [%s]", reason, self._call_id)
                    return
                self._enqueue_audio(audio)
                if not first_audio_ms:
                    first_audio_ms = (
                        time.perf_counter() - utterance.speech_end
                    ) * 1000.0
        tts_ms = (time.perf_counter() - tts_started) * 1000.0

        total_ms = (time.perf_counter() - utterance.speech_end) * 1000.0
        report = LatencyReport(
            call_id=self._call_id,
            text=content,
            asr_ms=asr_ms,
            llm_ms=llm_ms,
            tts_ms=tts_ms,
            total_ms=total_ms,
            queue_ms=queue_ms,
            first_audio_ms=first_audio_ms,
            llm_rounds_ms=tuple(llm_stats.get("rounds_ms") or ()),
            tool_ms=float(llm_stats.get("tool_ms") or 0.0),
            tts_chunks_ms=tuple(tts_chunks_ms),
            tts_first_chunk_ms=tts_chunks_ms[0] if tts_chunks_ms else 0.0,
        )
        with self._lock:
            self._reports.append(report)
        logger.info(
            "Задержка ответа [%s]: очередь %.2f с, ASR %.2f с, LLM %.2f с, "
            "TTS %.2f с, первый звук %.2f с, всего от конца речи %.2f с",
            self._call_id,
            queue_ms / 1000.0,
            asr_ms / 1000.0,
            llm_ms / 1000.0,
            tts_ms / 1000.0,
            first_audio_ms / 1000.0,
            total_ms / 1000.0,
        )
        logger.info(
            "Детали задержек [%s]: LLM раунды=[%s] мс, тулы=[%s] мс, "
            "TTS предложения=[%s] мс",
            self._call_id,
            ", ".join(f"{value:.1f}" for value in report.llm_rounds_ms),
            ", ".join(
                f"{value:.1f}" for value in (llm_stats.get("tools_ms") or ())
            ),
            ", ".join(f"{value:.1f}" for value in report.tts_chunks_ms),
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
        sentences = re.split(r"(?<=[.!?…])\s+", text)
        limit_sentences = self.config.max_reply_sentences
        if len(sentences) > limit_sentences:
            text = " ".join(sentences[:limit_sentences]).strip()
        limit = self.config.max_reply_chars
        if len(text) <= limit:
            return text

        cut = text[:limit]
        sentence_end = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        if sentence_end >= limit // 2:
            return cut[: sentence_end + 1].strip()
        space = cut.rfind(" ")
        return (cut[:space] if space > 0 else cut).strip()

    @staticmethod
    def _reply_chunks(text: str) -> List[str]:
        """Разбить ответ на предложения для потокового TTS.

        Короткие реплики вроде «Да.» приклеиваем к следующему предложению,
        чтобы TTS не делал неестественную паузу между ними.
        """
        parts = [
            part.strip()
            for part in re.split(r"(?<=[.!?…])\s+", text)
            if part.strip()
        ]
        chunks: List[str] = []
        for part in parts:
            if chunks and len(chunks[-1]) < 15:
                chunks[-1] = f"{chunks[-1]} {part}"
            else:
                chunks.append(part)
        return chunks or [text]

    def _publish(self, role: str, text: str) -> None:
        if self.store is None or not text:
            return
        try:
            self.store.publish(role, text, call_id=self._call_id)
        except Exception:  # noqa: BLE001 - UI не должен рвать звонок
            logger.exception("Не удалось опубликовать сообщение в UI")


__all__ = [
    "ASR_HALLUCINATION_PHRASES",
    "BILLING_TOOL",
    "DEFAULT_END_CALL_REPLY",
    "DEFAULT_FALLBACK",
    "DEFAULT_GREETING",
    "DEFAULT_NO_INPUT_PROMPT",
    "DialogBot",
    "DialogConfig",
    "END_CALL_TOOL",
    "LatencyReport",
]

