#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

runtime_source=ops/runtime_instance/local_benchmark.mbt
evidence_source=ops/runtime_instance/local_benchmark_evidence.mbt
types_source=ops/runtime_instance/local_benchmark_types.mbt
spawned_source=ops/runtime_instance/spawned_benchmark.mbt
cli_source=cmd/lunaflux/native_run.mbt
dispatch_source=cmd/lunaflux/main.mbt

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

for source_file in \
  "$runtime_source" \
  "$evidence_source" \
  "$types_source" \
  "$spawned_source" \
  "$cli_source" \
  "$dispatch_source"
do
  line_count=$(wc -l < "$source_file" | tr -d ' ')
  if [ "$line_count" -gt 500 ]; then
    fail "local benchmark source exceeds 500 lines: $source_file"
  fi
done

for anchor in \
  'fn exact_opaque_bench_argument(arguments : ArrayView[String])' \
  '[_, "bench", deployment] => Some(deployment)' \
  'Some(deployment) => run_native_benchmark(deployment)' \
  '@runtime_instance.benchmark_local_deployment(' \
  'benchmark_local_deployment(' \
  'let release = owner.release_admission' \
  'BenchmarkTrialCollector::new(' \
  'collector.submit(' \
  'collector.admit(' \
  'collector.first_token(' \
  'collector.terminate(' \
  'startup.readiness() != RuntimeReady' \
  'raw_events_canonical_hex=' \
  'summary_canonical_hex=' \
  'terminal_phase=closed' \
  'terminal_readiness=false' \
  'cleanup_complete=true' \
  'comparison_admission=not-run' \
  'promotion_authority=absent'
do
  if ! rg -F -q "$anchor" \
    "$runtime_source" "$evidence_source" "$spawned_source" \
    "$cli_source" "$dispatch_source"; then
    fail "local benchmark invariant is missing: $anchor"
  fi
done

# The measured owner and its release receipt must come from one deployment
# admission. A second preflight read could otherwise race the live prepare.
if rg -n 'preflight_release\(' "$runtime_source"; then
  fail 'local benchmark reparses deployment identity outside the live owner'
fi

# Local timings come only from the live monotonic collector. Captured/caller
# timestamps and the external comparison campaign are separate authorities.
if rg -n \
  'new_captured|submit_captured|admit_captured|first_token_captured|terminate_captured|CapturedMonotonicTimestamp|from_nanoseconds|BenchmarkComparison|openai_comparison|approved_model.*campaign' \
  "$runtime_source" "$evidence_source" "$spawned_source" "$cli_source"; then
  fail 'local benchmark gained caller timing or external comparison authority'
fi

if ! rg -q \
  'workload=fixed-token-0,greedy,streaming,cache-disabled,input-1,output-2' \
  "$evidence_source"; then
  fail 'local benchmark workload is no longer fixed and bounded'
fi

if ! rg -U -q \
  'if owner\.phase\(\) != Closed \|\|\n    owner\.readiness\(\) != RuntimeNotReady \|\|\n    !owner\.cleanup_complete\(\)' \
  "$runtime_source"; then
  fail 'local benchmark no longer checks terminal drain state'
fi

moon info --target native ops/runtime_instance cmd/lunaflux
moon check \
  --target native --deny-warn --warn-list +73 \
  ops/runtime_instance cmd/lunaflux
moon test \
  --target native --deny-warn --warn-list +73 \
  ops/runtime_instance cmd/lunaflux

printf '%s\n' 'LunaFlux bounded local benchmark CLI boundary is valid.'
