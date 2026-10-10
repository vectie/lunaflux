#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec moon run scripts/validate-token-step-performance-boundaries.mbtx "$@"
