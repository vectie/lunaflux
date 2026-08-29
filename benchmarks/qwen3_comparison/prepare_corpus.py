#!/usr/bin/env python3
"""Render Qwen chat inputs once and publish exact token IDs for all engines."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.qwen3_comparison.contract import (  # noqa: E402
    ContractError,
    PROFILE_CLASSES,
    canonical_json_bytes,
    read_digest_suffixed,
    require_sha256,
    sha256_bytes,
)


def _file_sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"pinned file is unavailable: {path.name}")
    return sha256_bytes(path.read_bytes())


def _load_sources(payload: bytes) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for line_number, raw in enumerate(payload.splitlines(), 1):
        if not raw:
            raise ContractError("message source contains an empty line")
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ContractError(f"message source line {line_number} is invalid") from error
        if not isinstance(row, dict) or set(row) != {
            "schema",
            "case_id",
            "profile_class",
            "messages",
        }:
            raise ContractError("message source fields are not exact")
        if row["schema"] != "lunaflux.qwen3-chat-case.v1":
            raise ContractError("message source schema mismatch")
        case_id = row["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ContractError("message case IDs must be unique")
        seen.add(case_id)
        if row["profile_class"] not in PROFILE_CLASSES:
            raise ContractError("message profile class is invalid")
        messages = row["messages"]
        if not isinstance(messages, list) or not messages:
            raise ContractError("message list must be nonempty")
        for message in messages:
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ContractError("chat message fields are not exact")
            if message["role"] not in ("system", "user", "assistant"):
                raise ContractError("chat message role is invalid")
            if not isinstance(message["content"], str):
                raise ContractError("chat message content must be text")
        rows.append(row)
    counts = {profile_class: 0 for profile_class in PROFILE_CLASSES}
    for row in rows:
        counts[row["profile_class"]] += 1
    if any(count < 32 for count in counts.values()):
        raise ContractError("at least 32 source chats are required per profile class")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--tokenizer-json-sha256", required=True)
    parser.add_argument("--tokenizer-config-sha256", required=True)
    parser.add_argument("--chat-template-sha256", required=True)
    parser.add_argument("--messages", required=True, help="ABSOLUTE_JSONL#sha256=HEX")
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        for label in (
            "config_sha256",
            "tokenizer_json_sha256",
            "tokenizer_config_sha256",
            "chat_template_sha256",
        ):
            require_sha256(getattr(arguments, label), label)
        root = arguments.model_root
        if not root.is_absolute() or not root.is_dir() or root.is_symlink() or root.resolve() != root:
            raise ContractError("model root must be a canonical absolute directory")
        expected_files = {
            "config.json": arguments.config_sha256,
            "tokenizer.json": arguments.tokenizer_json_sha256,
            "tokenizer_config.json": arguments.tokenizer_config_sha256,
        }
        for relative, expected in expected_files.items():
            if _file_sha(root / relative) != expected:
                raise ContractError(f"pinned file digest mismatch: {relative}")
        _, source_bytes, _ = read_digest_suffixed(arguments.messages, "messages")
        sources = _load_sources(source_bytes)
        if not arguments.output.is_absolute() or arguments.output.exists():
            raise ContractError("output must be a new absolute file")
        if arguments.output.parent.resolve() != arguments.output.parent:
            raise ContractError("output parent must be canonical")
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ContractError("the corpus Conda environment lacks transformers") from error
        tokenizer = AutoTokenizer.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=False, use_fast=True
        )
        template = tokenizer.chat_template
        if not isinstance(template, str) or sha256_bytes(template.encode()) != arguments.chat_template_sha256:
            raise ContractError("chat template digest mismatch")
        stage = arguments.output.with_name("." + arguments.output.name + ".partial")
        if stage.exists():
            raise ContractError("corpus staging path already exists")
        try:
            with stage.open("xb") as output:
                for source in sources:
                    rendered = tokenizer.apply_chat_template(
                        source["messages"], tokenize=False, add_generation_prompt=True
                    )
                    direct = tokenizer.apply_chat_template(
                        source["messages"], tokenize=True, add_generation_prompt=True
                    )
                    encoded = tokenizer.encode(rendered, add_special_tokens=False)
                    token_ids = [int(value) for value in encoded]
                    direct_ids = [int(value) for value in direct]
                    if token_ids != direct_ids:
                        raise ContractError("Qwen chat rendering/tokenization paths disagree")
                    output.write(
                        canonical_json_bytes(
                            {
                                "schema": "lunaflux.qwen3-tokenized-request.v1",
                                "case_id": source["case_id"],
                                "profile_class": source["profile_class"],
                                "chat_render_sha256": sha256_bytes(rendered.encode()),
                                "input_token_ids": token_ids,
                                "input_token_ids_sha256": sha256_bytes(
                                    canonical_json_bytes(token_ids)
                                ),
                                "input_tokens": len(token_ids),
                            }
                        )
                    )
                output.flush()
                os.fsync(output.fileno())
            os.chmod(stage, 0o444)
            os.replace(stage, arguments.output)
        except Exception:
            if stage.exists():
                os.chmod(stage, 0o600)
                stage.unlink()
            raise
        print("workload_sha256=" + _file_sha(arguments.output))
    except (ContractError, json.JSONDecodeError) as error:
        print(f"Qwen3 corpus preparation rejected: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
