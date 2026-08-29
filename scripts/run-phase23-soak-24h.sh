#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  else
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}

if [[ "${LUNAFLUX_RUN_PHASE23_SOAK_24H:-}" != "1" ]]; then
  printf '%s\n' \
    'set LUNAFLUX_RUN_PHASE23_SOAK_24H=1 to run the mandatory 24-hour soak' >&2
  exit 1
fi

moon check tests/worker_service_e2e \
  --target native --deny-warn --warn-list +73
moon build tests/worker_service_e2e cmd/worker_echo \
  --target native --release --deny-warn --warn-list +73

soak_bin_dir="$(mktemp -d -t lunaflux-phase23-soak)"
cp \
  _build/native/release/build/tests/worker_service_e2e/worker_service_e2e.exe \
  "$soak_bin_dir/worker_service_e2e.exe"
cp \
  _build/native/release/build/cmd/worker_echo/worker_echo.exe \
  "$soak_bin_dir/worker_echo.exe"
policy_file=tests/worker_service_e2e/phase23_soak_policy.v3
cp "$policy_file" "$soak_bin_dir/phase23_soak_policy.v3"
printf '%s\n' "Phase 2/3 soak immutable binaries: $soak_bin_dir"

source_file=tests/worker_service_e2e/phase23_soak_evidence.mbt
expected_policy_digest=ab268f305c71658b53b9f8347eb77a79d1fd4c414fc01faa0492a48e3886751a
if [[ "$(sha256_file "$policy_file")" != "$expected_policy_digest" ]]; then
  printf '%s\n' 'Phase 2/3 v3 soak policy manifest digest drifted' >&2
  exit 1
fi
if ! rg -q \
  'const PHASE23_SOAK_DURATION_MILLIS : UInt64 = 86400000UL' \
  "$source_file" ||
  ! rg -q 'while observed_millis - start_millis < PHASE23_SOAK_DURATION_MILLIS' \
    "$source_file" ||
  ! rg -q 'end_millis - start_millis < PHASE23_SOAK_DURATION_MILLIS' \
    "$source_file" ||
  ! rg -q 'PHASE23_SOAK_WAVE_PERIOD_MILLIS : UInt64 = 2000UL' \
    "$source_file" ||
  ! rg -q 'PHASE23_SOAK_VERSION : String = "luna-phase23-soak-v3"' \
    "$source_file" ||
  ! rg -q \
    "PHASE23_SOAK_POLICY_DIGEST : String = \"$expected_policy_digest\"" \
    "$source_file" ||
  ! rg -q 'ledger.two_live_waves > 0UL' "$source_file" ||
  ! rg -q 'ledger.batched_waves > 0UL' "$source_file" ||
  ! rg -q 'ledger.backpressured_waves > 0UL' "$source_file" ||
  ! rg -q 'ledger.scheduled_cancel_waves > 0UL' "$source_file" ||
  ! rg -q 'ledger.attempted_cancel_waves > 0UL' "$source_file" ||
  ! rg -q 'ledger.accepted_cancel_waves > 0UL' "$source_file" ||
  ! rg -q 'ledger.cancelled_waves > 0UL' "$source_file" ||
  ! rg -Fq '(cycle + 1) % 17 == 0' "$source_file" ||
  ! rg -q 'ledger.cycles % 211 == 0' "$source_file" ||
  ! rg -q 'ledger.cycles % 257 == 0' "$source_file" ||
  ! rg -q 'ledger.cycles % 337 == 0' "$source_file" ||
  [[ "$(rg -o 'server, connection, ledger, malformed, true,' "$source_file" | wc -l | tr -d ' ')" -ne 2 ]]; then
  printf '%s\n' 'Phase 2/3 v3 24-hour soak policy boundary drifted' >&2
  exit 1
fi
if rg -n 'duration[_-]?(millis|hours)?[=:]|TEMP|diagnostic|debug_inspect' \
  tests/worker_service_e2e/online_session_e2e_entry.mbt "$source_file"; then
  printf '%s\n' 'Phase 2/3 soak exposes a duration override or diagnostic' >&2
  exit 1
fi

"$soak_bin_dir/worker_service_e2e.exe" \
  --soak-24h \
  "$soak_bin_dir/worker_echo.exe" \
  "$repo_root"
