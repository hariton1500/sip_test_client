"""HTTP-сервер отдельного редактора промптов.

Только стандартная библиотека: страница редактора и небольшой JSON API для
чтения/создания/сохранения файлов. Сервер по умолчанию слушает localhost.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import unquote, urlsplit

from .prompts import PromptError, PromptStore

logger = logging.getLogger(__name__)

_HTML_PATH = Path(__file__).with_name("prompt_ui") / "index.html"
_API_PREFIX = "/api/prompts/"
_MAX_REQUEST_BYTES = 3 * 1024 * 1024


class _PromptHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, store: PromptStore) -> None:
        self.store = store
        self.stop_event = threading.Event()
        super().__init__(address, handler)


class _PromptHandler(BaseHTTPRequestHandler):
    server_version = "SipbotPromptEditor/1.0"

    @property
    def _store(self) -> PromptStore:
        return self.server.store  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802 - API BaseHTTPRequestHandler
        path = self._path()
        try:
            if path in ("/", "/index.html"):
                self._serve_html()
            elif path == "/api/prompts":
                self._send_json(
                    {
                        "root": str(self._store.root),
                        "prompts": [item.to_dict() for item in self._store.list()],
                    }
                )
            elif path.startswith(_API_PREFIX):
                name = path[len(_API_PREFIX) :]
                self._send_json(self._store.read(name).to_dict())
            elif path == "/api/health":
                self._send_json({"status": "ok", "root": str(self._store.root)})
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self._send_json({"error": "not found"}, status=404)
        except PromptError as exc:
            self._send_error(exc)

    def do_POST(self) -> None:  # noqa: N802 - API BaseHTTPRequestHandler
        path = self._path()
        try:
            if path != "/api/prompts":
                self._send_json({"error": "not found"}, status=404)
                return
            body = self._read_json()
            document = self._store.create(
                str(body.get("name") or ""),
                str(body.get("content") or ""),
            )
            self._send_json(document.to_dict(), status=201)
        except PromptError as exc:
            self._send_error(exc)

    def do_PUT(self) -> None:  # noqa: N802 - API BaseHTTPRequestHandler
        path = self._path()
        try:
            if not path.startswith(_API_PREFIX):
                self._send_json({"error": "not found"}, status=404)
                return
            name = path[len(_API_PREFIX) :]
            body = self._read_json()
            if "content" not in body or not isinstance(body["content"], str):
                self._send_json({"error": "Поле content обязательно"}, status=400)
                return
            document = self._store.save(
                name,
                body["content"],
                base_mtime_ns=body.get("base_mtime_ns"),
                force=bool(body.get("force")),
            )
            self._send_json(document.to_dict())
        except PromptError as exc:
            self._send_error(exc)

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("PromptUI: " + fmt, *args)

    # ------------------------------------------------------------------
    def _path(self) -> str:
        return unquote(urlsplit(self.path).path)

    def _read_json(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise PromptError("Некорректный Content-Length") from exc
        if length <= 0:
            raise PromptError("Пустое тело запроса")
        if length > _MAX_REQUEST_BYTES:
            raise PromptError("Тело запроса слишком большое")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PromptError("Ожидается корректный JSON") from exc
        if not isinstance(payload, dict):
            raise PromptError("JSON должен быть объектом")
        return payload

    def _serve_html(self) -> None:
        try:
            body = _HTML_PATH.read_bytes()
        except OSError as exc:
            self._send_json(
                {"error": f"не найден prompt_ui/index.html: {exc}"}, status=500
            )
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, exc: PromptError) -> None:
        self._send_json(
            {"error": exc.message, "code": exc.code},
            status=exc.status,
        )

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class PromptEditorServer:
    """Редактор промптов в фоновом демон-потоке."""

    def __init__(
        self,
        store: PromptStore,
        host: str = "127.0.0.1",
        port: int = 8766,
    ) -> None:
        self.store = store
        self._httpd = _PromptHTTPServer((host, port), _PromptHandler, store)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="sipbot-prompt-editor",
            daemon=True,
        )
        self.host = host
        self.port = int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}/"

    def start(self) -> None:
        self._thread.start()
        logger.info("Редактор промптов: %s", self.url)

    def stop(self) -> None:
        self._httpd.stop_event.set()
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def wait(self) -> None:
        """Блокировать основной поток до остановки сервера."""
        self._httpd.stop_event.wait()


def start_prompt_editor(
    store: PromptStore,
    host: str = "127.0.0.1",
    port: int = 8766,
) -> PromptEditorServer:
    """Создать и запустить сервер. Бросает OSError, если порт занят."""
    server = PromptEditorServer(store, host=host, port=port)
    server.start()
    return server


__all__ = ["PromptEditorServer", "start_prompt_editor"]

