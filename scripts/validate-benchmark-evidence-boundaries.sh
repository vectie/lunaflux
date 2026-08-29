#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

failed=0

if [ ! -f benchmarks/evidence/moon.pkg ]; then
  printf '%s\n' 'benchmark evidence package is missing' >&2
  exit 1
fi

if rg -n \
  'vectie/lunaflux/|moonbitlang/core/(fs|process|net)|moonbitlang/x/(fs|process|http)' \
  benchmarks/evidence/moon.pkg; then
  printf '%s\n' \
    'benchmark evidence admission must not own runtime, filesystem, process, or network authority' >&2
  failed=1
fi

if [ ! -f benchmarks/evidence/pkg.generated.mbti ] ||
  ! rg -U -q \
    'pub struct BenchmarkDigest \{\n  // private fields\n\}' \
    benchmarks/evidence/pkg.generated.mbti; then
  printf '%s\n' \
    'benchmark digest must remain opaque and constructible only after validation' >&2
  failed=1
fi

for engine in LunaFlux Vllm Sglang; do
  if [ "$(rg -c "^  ${engine}$" benchmarks/evidence/enums.mbt)" -ne 1 ]; then
    printf '%s\n' "benchmark engine ${engine} is not declared exactly once" >&2
    failed=1
  fi
done

for profile in \
  Latency Chat LongPrefill DecodeHeavy PrefixRich PrefixCold Saturation Churn Mixed
do
  if [ "$(rg -c "^  ${profile}$" benchmarks/evidence/enums.mbt)" -ne 1 ]; then
    printf '%s\n' "benchmark profile ${profile} is not declared exactly once" >&2
    failed=1
  fi
done

for required in \
  'trials.length() != 81' \
  'let accounted = Int64::from_int(completed)' \
  'accounted != Int64::from_int(submitted)' \
  'CorrectnessNotEstablished' \
  'BiasedEngineOrder' \
  'raw_events == correctness' \
  'NonCanonicalEventOrder' \
  'SummaryIntegerOverflow' \
  'lunaflux.benchmark-request-events.v1' \
  'lunaflux.benchmark-trial-summary.v1' \
  'lunaflux.benchmark-comparison.v1'
do
  if ! rg -F -q "$required" benchmarks/evidence --glob '*.mbt'; then
    printf '%s\n' "benchmark evidence invariant is missing: ${required}" >&2
    failed=1
  fi
done

if rg -n \
  'throughput[_ -]?pass|performance[_ -]?pass|baseline[_ -]?beaten|outcome=.*pass' \
  benchmarks/evidence docs/BENCHMARKING.md; then
  printf '%s\n' 'benchmark contract must not claim unmeasured performance' >&2
  failed=1
fi

while IFS= read -r source_file; do
  line_count=$(wc -l < "$source_file" | tr -d ' ')
  if [ "$line_count" -gt 500 ]; then
    printf '%s: %s lines; benchmark evidence files must stay cohesive\n' \
      "$source_file" "$line_count" >&2
    failed=1
  fi
done <<EOF
$(rg --files benchmarks/evidence --glob '*.mbt' | sort)
EOF

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'Benchmark evidence boundaries are valid.'
