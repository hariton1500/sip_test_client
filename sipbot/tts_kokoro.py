"""Локальный TTS на модели kokoro-ru (ONNX, CPU).

Модель `zaakirio/kokoro-ru` — русский порт Kokoro-82M: 82M параметров,
24 кГц, голоса ``sveta``/``masha``/``dima``. Тяжёлые зависимости (onnxruntime,
ruaccent, phonemizer, misaki) нужны только при включённом провайдере, поэтому
импортируются лениво: обычный облачный TTS продолжает работать без них.

Модуль отдаёт WAV bytes через тот же интерфейс, что и ``AiClient.synthesize``,
поэтому ``TtsService`` сам делает ресемплинг в 8 кГц, обрезку тишины и кэш.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import types
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .ru_text import normalize_for_tts
from .tts import pcm_s16_to_wav_bytes

logger = logging.getLogger(__name__)

#: Частота модели; TtsService пересэмплирует в 8 кГц для телефонии.
MODEL_SAMPLE_RATE = 24000
#: Kokoro принимает до 510 позиций, включая BOS/EOS; стиль — по длине ipa.
MAX_PHONEMES = 508
#: Максимальная длина текста на один проход (с запасом под фонемы).
MAX_CHUNK_CHARS = 300
#: Пауза между склеенными фрагментами длинного текста.
CHUNK_GAP_MS = 60

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_WORD_SPLIT = re.compile(r"\s+")


class KokoroUnavailable(RuntimeError):
    """Локальный TTS недоступен: нет зависимостей, модели или данных espeak."""


def _env_str(environ: Mapping[str, str], name: str, default: str) -> str:
    return str(environ.get(name, default) or default).strip()


def _env_int(
    environ: Mapping[str, str], name: str, default: int, minimum: int = 1
) -> int:
    raw = str(environ.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("Некорректное %s=%r, использую %d", name, raw, default)
        return default


def _env_float(
    environ: Mapping[str, str], name: str, default: float, minimum: float = 0.0
) -> float:
    raw = str(environ.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        logger.warning("Некорректное %s=%r, использую %.2f", name, raw, default)
        return default


def _env_bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    raw = str(environ.get(name, "") or "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    logger.warning("Некорректное %s=%r, использую %s", name, raw, default)
    return default


def _split_long_unit(unit: str, limit: int) -> List[str]:
    """Разбить слишком длинное предложение по словам."""
    words = [part for part in _WORD_SPLIT.split(unit) if part]
    pieces: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            pieces.append(current)
        current = word
    if current:
        pieces.append(current)
    return pieces or [unit[:limit]]


def split_text_for_tts(text: str, limit: int = MAX_CHUNK_CHARS) -> List[str]:
    """Нарезать текст на куски, не разрывая слова и по возможности предложения."""
    normalized = " ".join((text or "").split())
    if not normalized:
        return []
    if len(normalized) <= limit:
        return [normalized]

    units: List[str] = []
    for sentence in _SENTENCE_SPLIT.split(normalized):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= limit:
            units.append(sentence)
        else:
            units.extend(_split_long_unit(sentence, limit))

    chunks: List[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip()
        if current and len(candidate) > limit:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _float_audio_to_s16(audio: np.ndarray) -> bytes:
    """Нормализовать float [-1, 1] в s16le, не давая клиппингу зашкалить."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return b""
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    samples = np.clip(samples, -1.0, 1.0)
    return np.rint(samples * 32767.0).astype("<i2").tobytes()


class _OnnxKokoroEngine:
    """Обёртка над ONNX-сессией и русским G2P."""

    def __init__(
        self,
        session: Any,
        vocab: Mapping[str, int],
        style: np.ndarray,
        g2p: Any,
        speed: float = 1.0,
    ) -> None:
        self.sample_rate = MODEL_SAMPLE_RATE
        self._session = session
        self._vocab = dict(vocab)
        self._style = np.asarray(style, dtype=np.float32)
        self._g2p = g2p
        self._speed = float(speed)
        self._input_names = [item.name for item in session.get_inputs()]

    def phonemize(self, text: str) -> Tuple[str, set]:
        result = self._g2p(text)
        if isinstance(result, tuple):
            ipa, oov = result
        else:  # защита от другой версии ru_g2p
            ipa, oov = result, set()
        return str(ipa or ""), set(oov or ())

    def _input_name(self, candidates: Sequence[str], index: int) -> str:
        for name in candidates:
            if name in self._input_names:
                return name
        if len(self._input_names) > index:
            return self._input_names[index]
        raise KokoroUnavailable(
            f"Неожиданные входы ONNX-модели: {self._input_names}"
        )

    def render(self, ipa: str) -> np.ndarray:
        ids = [self._vocab[char] for char in ipa if char in self._vocab]
        if not ids:
            raise ValueError("G2P не дал ни одной фонемы из словаря модели")
        if len(ids) > MAX_PHONEMES:
            raise ValueError(
                f"Слишком длинная фонемная последовательность: {len(ids)}"
            )

        style_index = min(max(len(ipa) - 1, 0), len(self._style) - 1)
        style_row = self._style[style_index][None, :]
        style_row = np.ascontiguousarray(style_row, dtype=np.float32)

        feeds = {
            self._input_name(("input_ids", "tokens", "text"), 0): np.asarray(
                [[0, *ids, 0]], dtype=np.int64
            ),
            self._input_name(("style", "ref_s", "voice"), 1): style_row,
            self._input_name(("speed",), 2): np.asarray(
                [self._speed], dtype=np.float32
            ),
        }
        output = self._session.run(None, feeds)[0]
        return np.asarray(output, dtype=np.float32)


class KokoroTts:
    """Локальный русский TTS с интерфейсом ``AiClient.synthesize``.

    Публичные атрибуты ``tts_model``/``tts_voice``/``tts_sample_rate`` нужны
    ``TtsService``: по первым двум строится ключ кэша, третий используется,
    если провайдер вернул raw PCM (здесь всегда WAV).
    """

    def __init__(
        self,
        model_dir: Path,
        voice: str = "sveta",
        model_file: str = "onnx/model_quantized.onnx",
        repo_id: str = "zaakirio/kokoro-ru",
        threads: int = 2,
        speed: float = 1.0,
        auto_download: bool = True,
        espeak_data: Optional[Path] = None,
        normalizer=normalize_for_tts,
        engine: Optional[Any] = None,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser()
        self.voice = (voice or "sveta").strip() or "sveta"
        self.model_file = (model_file or "onnx/model_quantized.onnx").strip()
        self.repo_id = (repo_id or "zaakirio/kokoro-ru").strip()
        self.threads = max(1, int(threads))
        self.speed = min(2.0, max(0.1, float(speed)))
        self.auto_download = bool(auto_download)
        self.espeak_data = (
            Path(espeak_data).expanduser()
            if espeak_data
            else self.model_dir / "espeak-data"
        )
        self._normalizer = normalizer
        self._engine = engine
        self._lock = threading.Lock()

        self.tts_model = f"kokoro-ru:{Path(self.model_file).stem}"
        self.tts_voice = self.voice
        self.tts_sample_rate = MODEL_SAMPLE_RATE
        self.tts_response_format = "wav"
        self.tts_timeout = 60.0

    @classmethod
    def from_env(
        cls, environ: Optional[Mapping[str, str]] = None
    ) -> "KokoroTts":
        """Собрать провайдер из окружения и сразу загрузить модель."""
        source: Mapping[str, str] = os.environ if environ is None else environ
        instance = cls(
            model_dir=Path(
                _env_str(source, "KOKORO_MODEL_DIR", "cache/kokoro-ru")
            ),
            voice=_env_str(source, "KOKORO_VOICE", "sveta"),
            model_file=_env_str(
                source, "KOKORO_MODEL_FILE", "onnx/model_quantized.onnx"
            ),
            repo_id=_env_str(source, "KOKORO_REPO", "zaakirio/kokoro-ru"),
            threads=_env_int(source, "KOKORO_THREADS", 2),
            speed=_env_float(source, "KOKORO_SPEED", 1.0, minimum=0.1),
            auto_download=_env_bool(source, "KOKORO_DOWNLOAD", True),
            espeak_data=(
                Path(_env_str(source, "KOKORO_ESPEAK_DATA", "")).expanduser()
                if _env_str(source, "KOKORO_ESPEAK_DATA", "")
                else None
            ),
        )
        instance.load()
        return instance

    # ------------------------------------------------------------------
    # Загрузка модели
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Загрузить ONNX-сессию и G2P (идемпотентно)."""
        with self._lock:
            if self._engine is None:
                self._engine = self._create_engine()

    def warmup(self) -> None:
        """Прогреть модель короткой фразой (первый запуск самый долгий)."""
        self.synthesize("Здравствуйте")

    def close(self) -> None:
        with self._lock:
            self._engine = None

    def _required_files(self) -> List[Path]:
        return [
            self.model_dir / "config.json",
            self.model_dir / "kokoro-config.json",
            self.model_dir / "ru_g2p.py",
            self.model_dir / self.model_file,
            self.model_dir / "voices" / f"{self.voice}.bin",
            self.espeak_data / "ru_dict",
        ]

    def _ensure_files(self) -> None:
        missing = [path for path in self._required_files() if not path.is_file()]
        if not missing:
            return
        if not self.auto_download:
            names = ", ".join(str(path) for path in missing[:3])
            raise KokoroUnavailable(
                "Нет файлов модели kokoro-ru: "
                f"{names}. Включите KOKORO_DOWNLOAD=1 или положите модель "
                f"в {self.model_dir}"
            )
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise KokoroUnavailable(
                "Для загрузки kokoro-ru нужен пакет huggingface_hub: "
                "pip install -r requirements-kokoro.txt"
            ) from exc

        logger.info(
            "Загружаю kokoro-ru (%s, голос %s) в %s",
            self.repo_id,
            self.voice,
            self.model_dir,
        )
        patterns = [
            "config.json",
            "kokoro-config.json",
            "ru_g2p.py",
            self.model_file.replace("\\", "/"),
            f"voices/{self.voice}.bin",
            "espeak-data/*",
            "espeak-data/**",
        ]
        try:
            snapshot_download(
                repo_id=self.repo_id,
                local_dir=str(self.model_dir),
                allow_patterns=patterns,
            )
        except Exception as exc:  # noqa: BLE001 - сеть/HF отдают разные ошибки
            raise KokoroUnavailable(
                f"Не удалось скачать kokoro-ru: {exc}"
            ) from exc

        missing = [path for path in self._required_files() if not path.is_file()]
        if missing:
            raise KokoroUnavailable(
                "После загрузки не хватает файлов: "
                + ", ".join(str(path) for path in missing)
            )

    @staticmethod
    def _load_module(path: Path, name: str) -> Any:
        """Загрузить ``ru_g2p.py`` с исправлением кодировки.

        На Windows ``Path.read_text()`` без ``encoding`` использует cp1251, а
        ``config.json`` содержит UTF-8 IPA-символы. В результате G2P падает с
        ``'charmap' codec can't decode byte ...`` ещё до загрузки ONNX. Патчим
        только этот вызов в исходнике, не меняя файл на диске.
        """
        module = sys.modules.get(name)
        if module is not None:
            return module
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise KokoroUnavailable(f"Не удалось прочитать модуль {path}: {exc}") from exc
        source = source.replace(
            ".read_text()", ".read_text(encoding='utf-8')"
        )

        module = types.ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module
        try:
            code = compile(source, str(path), "exec")
            exec(code, module.__dict__)  # noqa: S102 - доверенный файл модели
        except Exception:
            sys.modules.pop(name, None)
            raise
        return module

    @staticmethod
    def _read_vocab(path: Path) -> dict:
        import json

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise KokoroUnavailable(f"Не удалось прочитать {path}: {exc}") from exc
        vocab = data.get("vocab") if isinstance(data, dict) else None
        if not isinstance(vocab, dict) or not vocab:
            raise KokoroUnavailable(f"В {path} нет словаря vocab")
        return {str(char): int(index) for char, index in vocab.items()}

    def _create_engine(self) -> _OnnxKokoroEngine:
        # Сначала проверяем зависимости: иначе при их отсутствии успеем
        # скачать сотни мегабайт модели и только потом упадём.
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise KokoroUnavailable(
                "Локальный TTS требует onnxruntime: "
                "pip install -r requirements-kokoro.txt"
            ) from exc

        self._ensure_files()

        # config.json — тот же словарь, который использует ru_g2p.
        vocab = self._read_vocab(self.model_dir / "config.json")

        try:
            module = self._load_module(
                self.model_dir / "ru_g2p.py", "_sipbot_kokoro_ru_g2p"
            )
            g2p = module.RuG2P(
                espeak_data=self.espeak_data,
                vocab_path=self.model_dir / "config.json",
            )
        except Exception as exc:  # noqa: BLE001 - ruaccent/misaki/espeak
            logger.debug("Ошибка G2P kokoro-ru", exc_info=True)
            raise KokoroUnavailable(
                "Не удалось инициализировать русский G2P kokoro-ru "
                f"(проверьте requirements-kokoro.txt и espeak-ng): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        style_path = self.model_dir / "voices" / f"{self.voice}.bin"
        style = np.fromfile(style_path, dtype=np.float32)
        if style.size == 0 or style.size % 256:
            raise KokoroUnavailable(
                f"Повреждён голосовой файл {style_path}: {style.size} float32"
            )
        style = style.reshape(-1, 256)

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        try:
            session = ort.InferenceSession(
                str(self.model_dir / self.model_file),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:  # noqa: BLE001 - ONNX сообщает разные типы
            raise KokoroUnavailable(
                f"Не удалось загрузить ONNX-модель kokoro-ru: {exc}"
            ) from exc

        logger.info(
            "kokoro-ru готов: голос=%s модель=%s threads=%d",
            self.voice,
            self.model_file,
            self.threads,
        )
        return _OnnxKokoroEngine(
            session=session,
            vocab=vocab,
            style=style,
            g2p=g2p,
            speed=self.speed,
        )

    # ------------------------------------------------------------------
    # Синтез
    # ------------------------------------------------------------------
    def synthesize(self, text: str) -> bytes:
        """Синтезировать текст и вернуть WAV (24 кГц, моно, s16)."""
        raw = " ".join((text or "").split())
        if not raw:
            return b""
        normalized = (self._normalizer(raw) or "").strip()
        if not normalized:
            raise ValueError("После нормализации текст для TTS пуст")

        self.load()
        engine = self._engine
        if engine is None:  # pragma: no cover - защита от гонки close()
            raise KokoroUnavailable("Модель kokoro-ru не загружена")

        gap = b"\x00\x00" * int(engine.sample_rate * CHUNK_GAP_MS / 1000)
        parts: List[bytes] = []
        # G2P (RUAccent + espeak) не потокобезопасен, а параллельные сессии
        # ONNX на CPU только мешают друг другу — синтез сериализуем.
        with self._lock:
            for chunk in split_text_for_tts(normalized):
                ipa, oov = engine.phonemize(chunk)
                if oov:
                    logger.warning(
                        "kokoro-ru: символы вне словаря %s в %r",
                        "".join(sorted(oov))[:20],
                        chunk[:80],
                    )
                audio = engine.render(ipa)
                pcm = _float_audio_to_s16(audio)
                if pcm:
                    parts.append(pcm)

        if not parts:
            raise ValueError("kokoro-ru вернул пустой звук")
        pcm = gap.join(parts) if len(parts) > 1 else parts[0]
        return pcm_s16_to_wav_bytes(pcm, engine.sample_rate)


__all__ = [
    "KokoroTts",
    "KokoroUnavailable",
    "split_text_for_tts",
]
