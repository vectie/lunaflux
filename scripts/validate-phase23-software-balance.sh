#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon check tests/worker_service_e2e \
  --target native --deny-warn --warn-list +73
moon build tests/worker_service_e2e cmd/worker_echo \
  --target native --release --deny-warn --warn-list +73

source_files=(
  tests/worker_service_e2e/phase23_balance_fixture.mbt
  tests/worker_service_e2e/phase23_balance_evidence.mbt
)
fixture_root=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-phase23-balance.XXXXXX")
fixture_root=$(CDPATH= cd -- "$fixture_root" && pwd -P)
cleanup() { rmdir -- "$fixture_root"; }
trap cleanup EXIT
if ! rg -q 'const PHASE23_BALANCE_REQUEST_COUNT : Int = 10000' "${source_files[@]}" ||
  ! rg -q 'max_active_requests=4' "${source_files[@]}" ||
  ! rg -q 'max_waiting_requests=8' "${source_files[@]}" ||
  ! rg -q 'saw_completion_backpressure' "${source_files[@]}" ||
  ! rg -q 'begin_shutdown_maintenance' "${source_files[@]}"; then
  printf '%s\n' 'Phase 2/3 balance evidence boundary drifted' >&2
  exit 1
fi
if rg -n 'TEMP|diagnostic|debug_inspect' "${source_files[@]}"; then
  printf '%s\n' 'Phase 2/3 balance evidence contains temporary diagnostics' >&2
  exit 1
fi

_build/native/release/build/tests/worker_service_e2e/worker_service_e2e.exe \
  --balance-10000 \
  _build/native/release/build/cmd/worker_echo/worker_echo.exe \
  "$fixture_root"

printf '%s\n' 'LunaFlux Phase 2/3 software balance gate passed.'
