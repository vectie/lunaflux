#!/usr/bin/env bash
set -eu

failed=0
fail() {
  printf '%s\n' "$1" >&2
  failed=1
}

runtime='ops/runtime_instance/spawned_benchmark.mbt'
types='ops/runtime_instance/spawned_benchmark_types.mbt'
campaign='tests/approved_model_spawned_physical/benchmark_campaign.mbt'
tests='tests/approved_model_spawned_physical/benchmark_campaign_wbtest.mbt'

for required in \
  '@benchmark_runner.BenchmarkTrialCollector::new' \
  'engine=LunaFlux' \
  'request_capacity=1' \
  'collector.submit' \
  'write_spawned_serving_frame' \
  'collector.admit' \
  'collector.first_token' \
  'collector.terminate' \
  'collector.finish' \
  'drain_spawned_serving_owner' \
  'spawned_serving_closed_state_valid'; do
  if ! rg -Fq "$required" "$runtime"; then
    fail "physical benchmark runtime adapter lost boundary: $required"
  fi
done

submit_line=$(rg -n 'collector\.submit' "$runtime" | cut -d: -f1)
write_line=$(rg -n 'write_spawned_serving_frame' "$runtime" | cut -d: -f1)
admit_line=$(rg -n 'collector\.admit' "$runtime" | cut -d: -f1)
first_line=$(rg -n 'collector\.first_token' "$runtime" | cut -d: -f1)
terminal_line=$(rg -n 'collector\.terminate' "$runtime" | cut -d: -f1)
finish_line=$(rg -n 'collector\.finish' "$runtime" | cut -d: -f1)
if [ -z "$submit_line" ] || [ -z "$write_line" ] || [ -z "$admit_line" ] ||
  [ -z "$first_line" ] || [ -z "$terminal_line" ] || [ -z "$finish_line" ] ||
  [ "$submit_line" -ge "$write_line" ] || [ "$write_line" -ge "$admit_line" ] ||
  [ "$admit_line" -ge "$first_line" ] || [ "$first_line" -ge "$terminal_line" ] ||
  [ "$terminal_line" -ge "$finish_line" ]; then
  fail 'physical benchmark lifecycle sampling order is not canonical'
fi

if rg -n '@socket|@hardware|@cuda|@device|@worker|@async/fs|@env' \
    benchmarks/runner --glob '*.mbt' --glob 'moon.pkg'; then
  fail 'bounded benchmark collector gained process/network/device authority'
fi

if rg -n 'now_millis|timestamp_at|let elapsed' "$campaign"; then
  fail 'physical campaign inferred or rewrote runner timestamps'
fi

for required in \
  'schema=lunaflux-physical-benchmark-trial.v1' \
  'measurement_sha256=' \
  'raw_events_sha256=' \
  'summary_sha256=' \
  'raw_events_canonical_hex=' \
  'summary_canonical_hex=' \
  'workload_contract=pinned-1-input-2-output-qualification' \
  'timing_scope=native-listener-request-only' \
  'clock_resolution_ns=1000000' \
  'correctness_scope=pinned-two-token-serving-contract' \
  'performance_scope=measured-qualification-not-baseline-comparison' \
  'comparison_admission=not-run' \
  'cleanup_complete=1' \
  'physical_benchmark_sha256='; do
  if ! rg -Fq "$required" "$campaign"; then
    fail "physical benchmark evidence lost canonical field: $required"
  fi
done

if rg -n 'comparison_admission=pass|correctness_scope=full|throughput_claim=pass' \
    "$campaign"; then
  fail 'single-request physical trial overclaims comparison or correctness'
fi

if ! rg -Fq 'physical benchmark canonical artifact storage is jointly bounded' \
    "$tests" ||
  ! rg -Fq 'physical benchmark scope cannot imply comparison admission' \
    "$tests" ||
  ! rg -Fq 'spawned benchmark contract fixes LunaFlux trial coordinates' \
    ops/runtime_instance/spawned_benchmark_wbtest.mbt; then
  fail 'physical benchmark hostile gates are incomplete'
fi

while IFS= read -r file; do
  lines=$(wc -l < "$file" | tr -d ' ')
  if [ "$lines" -ge 500 ]; then
    fail "physical benchmark source exceeds file budget: $file ($lines)"
  fi
done < <(printf '%s\n' "$runtime" "$types" "$campaign" "$tests" | sort)

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'physical benchmark adapter boundary gate passed'
