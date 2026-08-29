#!/usr/bin/env python3
"""Verify a campaign-local model admission without rescanning model weights."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.qwen3_comparison.contract import ContractError  # noqa: E402
from benchmarks.qwen3_comparison.lifecycle import validate_model_admission  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    try:
        validate_model_admission(sys.argv[2], Path(sys.argv[1]))
    except (ContractError, OSError):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
