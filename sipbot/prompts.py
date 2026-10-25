"""Безопасное чтение и редактирование файлов промптов.

Хранилище работает только с плоским каталогом (по умолчанию ``prompts/``),
разрешает текстовые расширения и не выпускает путь за пределы корня. Запись
атомарная: редактор не оставит наполовину записанный промпт при сбое.
"""

from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

ALLOWED_SUFFIXES = (".md", ".txt", ".prompt")
DEFAULT_MAX_BYTES = 2 * 1024 * 1024


class PromptError(Exception):
    """Ошибка операции с промптом, которую можно показать пользователю."""

    status = 400
    code = "bad_request"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class PromptNotFound(PromptError):
    status = 404
    code = "not_found"

    def __init__(self, name: str) -> None:
        super().__init__(f"Промпт не найден: {name}")


class PromptConflict(PromptError):
    status = 409
    code = "conflict"

    def __init__(self, message: str) -> None:
        super().__init__(message)


class PromptTooLarge(PromptError):
    status = 413
    code = "too_large"

    def __init__(self, size: int, limit: int) -> None:
        super().__init__(f"Файл слишком большой: {size} байт (лимит {limit})")


@dataclass(frozen=True)
class PromptInfo:
    """Краткая информация о файле для списка в интерфейсе."""

    name: str
    size: int
    modified_ns: int
    modified_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "modified_ns": str(self.modified_ns),
            "modified_at": self.modified_at,
        }


@dataclass(frozen=True)
class PromptDocument:
    """Полное содержимое промпта."""

    name: str
    content: str
    size: int
    modified_ns: int
    modified_at: float

    def info(self) -> PromptInfo:
        return PromptInfo(
            name=self.name,
            size=self.size,
            modified_ns=self.modified_ns,
            modified_at=self.modified_at,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "content": self.content,
            "size": self.size,
            "modified_ns": str(self.modified_ns),
            "modified_at": self.modified_at,
        }


class PromptStore:
    """Плоское хранилище промптов с атомарной записью."""

    def __init__(
        self,
        root: Union[str, Path],
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise PromptError(f"Не является каталогом: {self.root}")

    # ------------------------------------------------------------------
    # Проверка путей
    # ------------------------------------------------------------------
    def _resolve(self, name: str) -> Path:
        name = (name or "").strip()
        suffix = Path(name).suffix.lower() if name else ""
        if (
            not name
            or name in (".", "..")
            or name.startswith(".")
            or "/" in name
            or "\\" in name
            or "\x00" in name
            or any(ord(char) < 32 for char in name)
            or suffix not in ALLOWED_SUFFIXES
            or len(name) > 120
        ):
            raise PromptError(
                "Имя файла должно быть простым и иметь расширение "
                + ", ".join(ALLOWED_SUFFIXES)
            )
        path = (self.root / name).resolve()
        if path.parent != self.root:
            raise PromptError("Путь выходит за пределы каталога промптов")
        return path

    @staticmethod
    def _coerce_mtime(value: Optional[Union[int, str]]) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise PromptError("Некорректная метка времени файла") from exc

    # ------------------------------------------------------------------
    # Чтение
    # ------------------------------------------------------------------
    def list(self) -> List[PromptInfo]:
        """Вернуть все поддерживаемые файлы промптов по алфавиту."""
        items: List[PromptInfo] = []
        for path in self.root.iterdir():
            if (
                not path.is_file()
                or path.name.startswith(".")
                or path.suffix.lower() not in ALLOWED_SUFFIXES
            ):
                continue
            try:
                info = path.stat()
            except OSError:
                continue
            items.append(
                PromptInfo(
                    name=path.name,
                    size=info.st_size,
                    modified_ns=info.st_mtime_ns,
                    modified_at=info.st_mtime,
                )
            )
        items.sort(key=lambda item: item.name.casefold())
        return items

    def read(self, name: str) -> PromptDocument:
        path = self._resolve(name)
        try:
            info = path.stat()
        except FileNotFoundError as exc:
            raise PromptNotFound(name) from exc
        except OSError as exc:
            raise PromptError(f"Не удалось прочитать {name}: {exc}") from exc
        if not path.is_file():
            raise PromptNotFound(name)
        if info.st_size > self.max_bytes:
            raise PromptTooLarge(info.st_size, self.max_bytes)
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise PromptError(f"Файл {name} не является текстом UTF-8") from exc
        except OSError as exc:
            raise PromptError(f"Не удалось прочитать {name}: {exc}") from exc
        return PromptDocument(
            name=path.name,
            content=content,
            size=info.st_size,
            modified_ns=info.st_mtime_ns,
            modified_at=info.st_mtime,
        )

    # ------------------------------------------------------------------
    # Запись
    # ------------------------------------------------------------------
    def create(self, name: str, content: str = "") -> PromptDocument:
        path = self._resolve(name)
        if path.exists():
            raise PromptConflict(f"Файл уже существует: {path.name}")
        self._validate_content(content)
        self._write_atomic(path, content)
        return self.read(path.name)

    def save(
        self,
        name: str,
        content: str,
        base_mtime_ns: Optional[Union[int, str]] = None,
        force: bool = False,
    ) -> PromptDocument:
        """Сохранить промпт, не затирая чужие изменения без подтверждения."""
        path = self._resolve(name)
        try:
            info = path.stat()
        except FileNotFoundError as exc:
            raise PromptNotFound(name) from exc
        if not path.is_file():
            raise PromptNotFound(name)
        self._validate_content(content)

        expected = self._coerce_mtime(base_mtime_ns)
        if expected is not None and not force and info.st_mtime_ns != expected:
            raise PromptConflict(
                "Файл был изменён на диске. Перезагрузите его или "
                "подтвердите перезапись."
            )

        self._write_atomic(path, content)
        return self.read(path.name)

    def _validate_content(self, content: str) -> None:
        if not isinstance(content, str):
            raise PromptError("Содержимое должно быть строкой")
        if "\x00" in content:
            raise PromptError("Содержимое не должно содержать NUL-байты")
        size = len(content.encode("utf-8"))
        if size > self.max_bytes:
            raise PromptTooLarge(size, self.max_bytes)

    @staticmethod
    def _write_atomic(path: Path, content: str) -> None:
        data = content.encode("utf-8")
        fd = -1
        tmp_name = ""
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=str(path.parent),
            )
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if path.exists():
                os.chmod(tmp_name, stat.S_IMODE(path.stat().st_mode))
            os.replace(tmp_name, path)
        except OSError as exc:
            raise PromptError(f"Не удалось сохранить {path.name}: {exc}") from exc
        finally:
            if fd >= 0:
                os.close(fd)
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass


__all__ = [
    "ALLOWED_SUFFIXES",
    "PromptConflict",
    "PromptDocument",
    "PromptError",
    "PromptInfo",
    "PromptNotFound",
    "PromptStore",
    "PromptTooLarge",
]

