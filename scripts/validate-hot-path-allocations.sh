#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon run --target native --release tests/hot_path_alloc
scripts/validate-device-step-allocations.sh
scripts/validate-device-worker-one-row-allocations.sh
scripts/validate-token-step-performance-boundaries.sh
scripts/validate-fp8-v3-frame-allocations.sh
