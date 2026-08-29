#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

runner_dir=benchmarks/runner
failed=0

fail() {
  printf '%s\n' "$1" >&2
  failed=1
}

if [ ! -f "$runner_dir/moon.pkg" ] ||
  [ ! -f "$runner_dir/collector.mbt" ] ||
  [ ! -f "$runner_dir/terminal.mbt" ]; then
  fail 'benchmark runner package is incomplete'
fi

if rg -n \
  'moonbitlang/core/(fs|process|net)|moonbitlang/x/(fs|process|http)|vectie/lunaflux/(api|cmd|device|engine|internal|kernels|ops|service)' \
  "$runner_dir/moon.pkg"; then
  fail 'benchmark runner gained filesystem, process, network, device, or service authority'
fi

for dependency in \
  'vectie/lunaflux/benchmarks/evidence' \
  'vectie/lunaflux/runtime/monotonic_clock'
do
  if [ "$(rg -c -F "$dependency" "$runner_dir/moon.pkg")" -ne 1 ]; then
    fail "benchmark runner dependency changed: ${dependency}"
  fi
done

public_api=$(rg -n '^pub(\(all\))? (struct|enum|suberror|fn)' \
  "$runner_dir" --glob '*.mbt' --glob '!**/*test.mbt' || true)
public_symbols=$(
  printf '%s\n' "$public_api" |
    sed -E -n \
      -e 's#^.*:pub(\(all\))? (struct|enum|suberror) ([A-Za-z0-9_]+).*#type \2 \3#p' \
      -e 's#^.*:pub fn ([^(]+)\(.*#fn \1#p' |
    LC_ALL=C sort
)
expected_public_symbols=$(
  cat <<'EOF' | LC_ALL=C sort
fn BenchmarkTrialCollector::admit
fn BenchmarkTrialCollector::admit_captured
fn BenchmarkTrialCollector::finish
fn BenchmarkTrialCollector::first_token
fn BenchmarkTrialCollector::first_token_captured
fn BenchmarkTrialCollector::new
fn BenchmarkTrialCollector::new_captured
fn BenchmarkTrialCollector::submit
fn BenchmarkTrialCollector::submit_captured
fn BenchmarkTrialCollector::terminate
fn BenchmarkTrialCollector::terminate_captured
fn CapturedMonotonicTimestamp::from_nanoseconds
fn captured_monotonic_nanos_measurement_digest
fn system_monotonic_millis_measurement_digest
type struct BenchmarkTrialCollector
type struct CapturedMonotonicTimestamp
type suberror BenchmarkRunnerError
EOF
)
if [ "$public_symbols" != "$expected_public_symbols" ]; then
  printf '%s\n%s\n' 'benchmark runner exposes an unreviewed public API:' \
    "$public_symbols" >&2
  failed=1
fi

for required in \
  'FixedArray::make(request_capacity, Vacant)' \
  '@evidence.maximum_request_events_per_summary()' \
  'checked_nonnegative_add' \
  'BenchmarkRunnerClockMovedBackward' \
  'self.submitted != self.request_capacity || self.terminal != self.submitted' \
  '@evidence.BenchmarkSummaryEvidence::admit(' \
  'db8b9983e81661e4c955ebaed2edf6c35ba789529fa88b2f02f8b7471ca2dd7b' \
  '5b0872f4a1379903204d1bf61b5f51a80af28c7c559634abcc1a988467ce4d4b' \
  'captured_mode: false' \
  'captured_mode: true' \
  'measurement: system_monotonic_millis_measurement_digest()' \
  'measurement: captured_monotonic_nanos_measurement_digest()' \
  'measurement=self.measurement' \
  'if self.captured_mode {' \
  'if !self.captured_mode {' \
  'captured collector is clock-branded and live ingress cannot cross modes'
do
  if ! rg -F -q "$required" "$runner_dir" --glob '*.mbt'; then
    fail "benchmark runner invariant is missing: ${required}"
  fi
done

if rg -n '^pub fn .*_at\(' "$runner_dir" --glob '*.mbt'; then
  fail 'caller-supplied timestamp entry point escaped the benchmark runner package'
fi

if rg -n \
  'Command|Process|Socket|Tcp|Cuda|DeviceContext|absolute_path|listen_address' \
  "$runner_dir" --glob '*.mbt'; then
  fail 'benchmark runner source gained ambient execution or transport vocabulary'
fi

if [ ! -f "$runner_dir/pkg.generated.mbti" ] ||
  ! rg -U -q \
    'pub struct BenchmarkTrialCollector \{\n  // private fields\n\}' \
    "$runner_dir/pkg.generated.mbti"; then
  fail 'benchmark trial collector generated interface is not opaque'
fi

if [ ! -f "$runner_dir/pkg.generated.mbti" ] ||
  ! rg -U -q \
    'pub struct CapturedMonotonicTimestamp \{\n  // private fields\n\}' \
    "$runner_dir/pkg.generated.mbti"; then
  fail 'captured monotonic timestamp generated interface is not opaque'
fi

while IFS= read -r source_file; do
  line_count=$(wc -l < "$source_file" | tr -d ' ')
  if [ "$line_count" -gt 500 ]; then
    printf '%s: %s lines; benchmark runner files must stay cohesive\n' \
      "$source_file" "$line_count" >&2
    failed=1
  fi
done <<EOF
$(rg --files "$runner_dir" --glob '*.mbt' | sort)
EOF

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'Benchmark runner boundaries are valid.'
