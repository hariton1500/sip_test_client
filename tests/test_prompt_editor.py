"""Тесты редактора промптов: хранилище и HTTP API (без браузера)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from sipbot.prompt_ui_server import PromptEditorServer
from sipbot.prompts import (
    PromptConflict,
    PromptError,
    PromptNotFound,
    PromptStore,
)


def _request(url: str, method: str = "GET", payload=None):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=3) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_store_crud_and_atomic_write(tmp_path: Path):
    root = tmp_path / "prompts"
    store = PromptStore(root)

    created = store.create("support.md", "# Поддержка\n")
    assert created.name == "support.md"
    assert (root / "support.md").read_text(encoding="utf-8") == "# Поддержка\n"
    assert [item.name for item in store.list()] == ["support.md"]

    updated = store.save(
        "support.md",
        "# Поддержка\n\nНовый текст.\n",
        base_mtime_ns=created.modified_ns,
    )
    assert updated.content.endswith("Новый текст.\n")
    assert updated.size == len(updated.content.encode("utf-8"))
    assert not list(root.glob(".*.tmp"))


def test_store_rejects_unsafe_or_unsupported_names(tmp_path: Path):
    store = PromptStore(tmp_path / "prompts")
    for name in ("", "../secret.md", "sub/prompt.md", "prompt.exe", ".hidden.md"):
        with pytest.raises(PromptError):
            store.read(name)

    with pytest.raises(PromptError):
        store.create("support.json", "{}")


def test_store_conflict_and_force(tmp_path: Path):
    root = tmp_path / "prompts"
    store = PromptStore(root)
    created = store.create("a.md", "old")

    path = root / "a.md"
    changed = created.modified_ns + 1_000_000
    os.utime(path, ns=(changed, changed))

    with pytest.raises(PromptConflict):
        store.save("a.md", "from editor", base_mtime_ns=created.modified_ns)

    saved = store.save(
        "a.md",
        "from editor",
        base_mtime_ns=created.modified_ns,
        force=True,
    )
    assert saved.content == "from editor"

    with pytest.raises(PromptNotFound):
        store.read("missing.md")


def test_http_api_roundtrip(tmp_path: Path):
    store = PromptStore(tmp_path / "prompts")
    server = PromptEditorServer(store, host="127.0.0.1", port=0)
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        with urllib.request.urlopen(base + "/", timeout=3) as response:
            html = response.read().decode("utf-8")
            assert "Редактор промптов" in html
            assert "renderMarkdown" in html

        status, data = _request(
            base + "/api/prompts",
            method="POST",
            payload={"name": "new.md", "content": "# Новый\n"},
        )
        assert status == 201
        assert data["name"] == "new.md"

        _, listing = _request(base + "/api/prompts")
        assert listing["root"] == str(store.root)
        assert [item["name"] for item in listing["prompts"]] == ["new.md"]
        mtime = listing["prompts"][0]["modified_ns"]

        _, document = _request(base + "/api/prompts/new.md")
        assert document["content"] == "# Новый\n"

        status, saved = _request(
            base + "/api/prompts/new.md",
            method="PUT",
            payload={
                "content": "# Новый\n\nТекст.\n",
                "base_mtime_ns": mtime,
            },
        )
        assert status == 200
        assert "Текст." in saved["content"]

        encoded = urllib.parse.quote("../outside.md", safe="")
        try:
            _request(base + "/api/prompts/" + encoded)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
        else:  # pragma: no cover - защита должна сработать
            raise AssertionError("path traversal должен быть отклонён")
    finally:
        server.stop()
