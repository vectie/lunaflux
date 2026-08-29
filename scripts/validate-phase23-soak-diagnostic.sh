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

entry=tests/worker_service_e2e/online_session_e2e_entry.mbt
production=tests/worker_service_e2e/phase23_soak_evidence.mbt
diagnostic=tests/worker_service_e2e/phase23_soak_diagnostic.mbt
diagnostic_test=tests/worker_service_e2e/phase23_soak_diagnostic_wbtest.mbt
early_cancel=tests/worker_service_e2e/luna_pipeline_early_cancel_evidence.mbt
policy=tests/worker_service_e2e/phase23_soak_policy.v3
expected_policy_digest=ab268f305c71658b53b9f8347eb77a79d1fd4c414fc01faa0492a48e3886751a

required_diagnostic_evidence=(
  'const PHASE23_SOAK_DIAGNOSTIC_CYCLES : Int = 2300'
  'phase23_soak_wave('
  'ledger.cycles % 211 == 0'
  'ledger.cycles % 337 == 0'
  'ledger.cycles % 257 == 0'
  '"client-write-before"'
  '"server-progress-before"'
  '"server-progress-exhausted"'
  'saw_two_live=\{saw_two_live}'
  'saw_two_row_plan=\{saw_two_row_plan}'
  'saw_backpressure=\{saw_backpressure}'
  'phase23_soak_record_balanced_wave('
  'two_live_waves'
  'batched_waves'
  'backpressured_waves'
  'scheduled_cancel_waves'
  'attempted_cancel_waves'
  'accepted_cancel_waves'
  'cancelled_waves'
  '"client-read-before"'
  '"quiescence-before"'
  '"wave-complete"'
  '"foreign-reconnect-before"'
  '"malformed-reconnect-before"'
  '"plain-reconnect-before"'
  '"timer-before"'
  '@async.sleep(1)'
  '"finish-before"'
)

for evidence in "${required_diagnostic_evidence[@]}"; do
  if ! rg -Fq "$evidence" "$diagnostic" "$production"; then
    printf 'missing Phase 2/3 soak diagnostic evidence: %s\n' "$evidence" >&2
    exit 1
  fi
done

if ! rg -Fq 'phase23 balanced non-batched scheduled cancel remains legal' "$diagnostic_test" ||
  ! rg -Fq 'ledger, true, false, true, true, true, false, false' \
    "$diagnostic_test" ||
  ! rg -Fq 'assert_eq(ledger.batched_waves, 0UL)' "$diagnostic_test"; then
  printf '%s\n' 'Phase 2/3 non-batched balance regression evidence drifted' >&2
  exit 1
fi

if ! rg -Fq 'multi_scheduler_blueprint_for_rows(1)' "$early_cancel" ||
  ! rg -Fq 'LunaOnlineTcpFailed(LunaOnlineTcpCoordinator) => false' \
    "$early_cancel" ||
  ! rg -Fq 'assert_true(terminal_rejected)' "$early_cancel" ||
  ! rg -Fq 'metrics.histogram_bucket_value(BatchRows, 1), 0UL' \
    "$early_cancel" ||
  ! rg -Fq 'progress_server_to_retired(server)' "$early_cancel"; then
  printf '%s\n' 'Phase 2/3 early-cancel integration evidence drifted' >&2
  exit 1
fi

if ! rg -Fq '[_, "--soak-cycle-fast", executable, fixture_root]' "$entry" ||
  ! rg -Fq '[_, "--soak-cycle-timer", executable, fixture_root]' "$entry" ||
  ! rg -Fq '[_, "--pipeline-early-cancel", executable]' "$entry"; then
  printf '%s\n' 'Phase 2/3 soak diagnostic entry drifted' >&2
  exit 1
fi

if ! rg -Fq 'prepare_pipeline_service(executable, fixture_root~)' \
  "$production" ||
  ! rg -Fq 'bootstrap_source_for_root(fixture_root)' \
    tests/worker_service_e2e/luna_pipeline_server_evidence.mbt ||
  ! rg -Fq 'tcp_service_binding(fixture_root~)' \
    tests/worker_service_e2e/luna_pipeline_server_evidence.mbt ||
  ! rg -Fq '"$repo_root"' scripts/run-phase23-soak-24h.sh; then
  printf '%s\n' 'Phase 2/3 soak portable approved-root wiring drifted' >&2
  exit 1
fi

if ! rg -Fq \
  'const PHASE23_SOAK_DURATION_MILLIS : UInt64 = 86400000UL' \
  "$production" ||
  ! rg -Fq \
    'const PHASE23_SOAK_WAVE_PERIOD_MILLIS : UInt64 = 2000UL' \
    "$production" ||
  ! rg -Fq \
    'const PHASE23_SOAK_VERSION : String = "luna-phase23-soak-v3"' \
    "$production" ||
  ! rg -Fq \
    "const PHASE23_SOAK_POLICY_DIGEST : String = \"$expected_policy_digest\"" \
    "$production"; then
  printf '%s\n' 'Phase 2/3 production soak boundary drifted' >&2
  exit 1
fi

if [[ "$(sha256_file "$policy")" != "$expected_policy_digest" ]] ||
  ! rg -Fxq 'schema=lunaflux.phase23-soak-policy.v1' "$policy" ||
  ! rg -Fxq 'version=luna-phase23-soak-v3' "$policy" ||
  ! rg -Fxq 'duration_millis=86400000' "$policy" ||
  ! rg -Fxq 'wave_period_millis=2000' "$policy" ||
  ! rg -Fxq 'cancel_period_cycles=17' "$policy" ||
  ! rg -Fxq 'foreign_rejection_period_cycles=211' "$policy" ||
  ! rg -Fxq 'malformed_rejection_period_cycles=337' "$policy" ||
  ! rg -Fxq 'reconnect_period_cycles=257' "$policy" ||
  ! rg -Fxq 'malformed_counts_as_rejection=1' "$policy" ||
  ! rg -Fxq \
    'required_cumulative_invariants=two_live_waves,batched_waves,backpressured_waves,scheduled_cancel_waves,attempted_cancel_waves,accepted_cancel_waves,cancelled_waves' \
    "$policy" ||
  [[ "$(rg -o 'server, connection, ledger, malformed, true,' "$production" "$diagnostic" | wc -l | tr -d ' ')" -ne 3 ]]; then
  printf '%s\n' 'Phase 2/3 v3 soak policy manifest or schedule drifted' >&2
  exit 1
fi

if rg -n 'duration[_-]?(millis|hours)?[=:]|debug_inspect' \
  "$entry" "$production"; then
  printf '%s\n' 'Phase 2/3 soak exposes a duration override' >&2
  exit 1
fi

printf '%s\n' 'Phase 2/3 soak diagnostic source gate passed.'
