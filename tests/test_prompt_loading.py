"""Тесты сборки system prompt из нескольких файлов."""

from __future__ import annotations

from pathlib import Path

import pytest

from sipbot.prompts import (
    DEFAULT_PROMPT_SEPARATOR,
    DEFAULT_SYSTEM_PROMPT_FILES,
    PromptError,
    PromptNotFound,
    PromptStore,
    parse_prompt_file_names,
    resolve_system_prompt_files,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(root: Path, name: str, text: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(text, encoding="utf-8")


def test_load_many_preserves_order_and_separator(tmp_path: Path):
    root = tmp_path / "prompts"
    _write(root, "a.md", "Первый")
    _write(root, "b.md", "Второй")
    store = PromptStore(root)

    bundle = store.load_many(
        ["b.md", "prompts/a.md", "b.md"],
        separator="\n--\n",
    )

    assert bundle.names == ("b.md", "a.md")
    assert bundle.content == "Второй\n--\nПервый"
    assert bundle.size == len(bundle.content.encode("utf-8"))
    assert bundle.to_dict() == {
        "names": ["b.md", "a.md"],
        "content": "Второй\n--\nПервый",
        "size": bundle.size,
    }


def test_default_bundle_contains_all_project_prompts():
    store = PromptStore(REPO_ROOT / "prompts")

    bundle = store.load_system_prompt()

    assert bundle.names == DEFAULT_SYSTEM_PROMPT_FILES
    assert DEFAULT_PROMPT_SEPARATOR in bundle.content
    for document in bundle.documents:
        assert document.content.strip()
    assert bundle.size > sum(len(item.content) for item in bundle.documents)


def test_load_many_reports_missing_file(tmp_path: Path):
    store = PromptStore(tmp_path / "prompts")
    _write(store.root, "exists.md", "ok")

    with pytest.raises(PromptNotFound, match="missing.md"):
        store.load_many(["exists.md", "missing.md"])


def test_load_many_rejects_empty_lists(tmp_path: Path):
    store = PromptStore(tmp_path / "prompts")

    for names in ([], (), ["", "  "], None):
        with pytest.raises(PromptError):
            store.load_many(names)  # type: ignore[arg-type]


def test_parse_prompt_file_names():
    assert parse_prompt_file_names(
        " prompts\\support_evp.md; connection_terms.md,\n"
        "connection_terms.md\r\ninternet_private.md "
    ) == (
        "support_evp.md",
        "connection_terms.md",
        "internet_private.md",
    )
    assert parse_prompt_file_names("") == ()
    with pytest.raises(PromptError):
        parse_prompt_file_names("a.md\x00b.md")


def test_resolve_system_prompt_files_precedence():
    both = {
        "SYSTEM_PROMPT_FILES": "a.md, b.md",
        "SYSTEM_PROMPT_FILE": "legacy.md",
    }
    assert resolve_system_prompt_files(both) == ("a.md", "b.md")

    legacy = {
        "SYSTEM_PROMPT_FILES": "  ",
        "SYSTEM_PROMPT_FILE": "prompts/legacy.md",
    }
    assert resolve_system_prompt_files(legacy) == ("legacy.md",)

    assert resolve_system_prompt_files({}) == DEFAULT_SYSTEM_PROMPT_FILES
    assert resolve_system_prompt_files(
        {"SYSTEM_PROMPT_FILES": "a.md; b.md"}
    ) == ("a.md", "b.md")


def test_resolve_rejects_empty_configured_list():
    with pytest.raises(PromptError):
        resolve_system_prompt_files({"SYSTEM_PROMPT_FILES": " , ; "})


def test_load_system_prompt_from_env(tmp_path: Path):
    root = tmp_path / "prompts"
    _write(root, "a.md", "A")
    _write(root, "b.md", "B")
    store = PromptStore(root)

    bundle = store.load_system_prompt_from_env(
        {"SYSTEM_PROMPT_FILES": "a.md,b.md"},
        separator=" | ",
    )

    assert bundle.names == ("a.md", "b.md")
    assert bundle.content == "A | B"
