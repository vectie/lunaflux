#!/usr/bin/env python3
"""Command adapter for exact source-model inventory admission."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.qwen3_comparison.contract import (  # noqa: E402
    ContractError,
    validate_model_inventory,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("sha256")
    arguments = parser.parse_args()
    try:
        validate_model_inventory(arguments.root, arguments.inventory, arguments.sha256)
    except ContractError as error:
        print(f"Qwen3 source-model inventory rejected: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
