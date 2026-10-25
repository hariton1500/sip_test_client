"""Локальный веб-интерфейс для живой ленты разговора.

Используется только стандартная библиотека: ``ThreadingHTTPServer`` отдаёт
одну HTML-страницу и поток Server-Sent Events (SSE). Новые зависимости не
нужны, а браузер можно открыть на этой же машине.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from .transcript import TranscriptStore

logger = logging.getLogger(__name__)

_HTML_PATH = Path(__file__).with_name("ui") / "index.html"


class _UiHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, store: TranscriptStore) -> None:
        self.store = store
        self.stop_event = threading.Event()
        super().__init__(address, handler)


class _UiHandler(BaseHTTPRequestHandler):
    server_version = "SipbotUI/1.0"

    @property
    def _store(self) -> TranscriptStore:
        return self.server.store  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802 - API BaseHTTPRequestHandler
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._serve_html()
        elif path == "/events":
            self._serve_events()
        elif path == "/api/messages":
            self._send_json({"messages": self._store.snapshot()})
        elif path == "/api/health":
            self._send_json(
                {"status": "ok", "listeners": self._store.listener_count}
            )
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - API BaseHTTPRequestHandler
        path = self.path.split("?", 1)[0]
        if path != "/api/clear":
            self._send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(min(length, 65536))
        self._store.clear()
        self._send_json({"status": "cleared"})

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("UI: " + fmt, *args)

    # ------------------------------------------------------------------
    def _serve_html(self) -> None:
        try:
            body = _HTML_PATH.read_bytes()
        except OSError as exc:
            self._send_json(
                {"error": f"не найден ui/index.html: {exc}"}, status=500
            )
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        listener = self._store.subscribe()
        stop_event = self.server.stop_event  # type: ignore[attr-defined]
        try:
            self._sse_send("snapshot", {"messages": self._store.snapshot()})
            while not stop_event.is_set():
                try:
                    event = listener.get(timeout=15.0)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                event_type = event.get("type", "message")
                data = {k: v for k, v in event.items() if k != "type"}
                self._sse_send(event_type, data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            self._store.unsubscribe(listener)
            self.close_connection = True

    def _sse_send(self, event: str, data: Dict[str, Any]) -> None:
        body = (
            f"event: {event}\n"
            f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        ).encode("utf-8")
        self.wfile.write(body)
        self.wfile.flush()

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class UiServer:
    """HTTP-сервер интерфейса в фоновом демон-потоке."""

    def __init__(
        self,
        store: TranscriptStore,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.store = store
        self._httpd = _UiHTTPServer((host, port), _UiHandler, store)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="sipbot-ui",
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
        logger.info("Веб-интерфейс разговора: %s", self.url)

    def stop(self) -> None:
        self._httpd.stop_event.set()
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


def start_ui_server(
    store: TranscriptStore,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> UiServer:
    """Создаёт и запускает UI. Бросает OSError, если порт занят."""
    server = UiServer(store, host=host, port=port)
    server.start()
    return server


__all__ = ["UiServer", "start_ui_server"]

