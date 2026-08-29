#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

moon run --target native --release tests/tensor_parallel_device_worker_alloc

generated_c="_build/native/release/build/tests/tensor_parallel_device_worker_alloc/tensor_parallel_device_worker_alloc.c"
if [ ! -f "$generated_c" ]; then
  printf '%s\n' 'tensor-parallel device-worker release C output is missing' >&2
  exit 1
fi

# The dynamic counters prove execution. These retained symbols additionally
# make the evidence scope reviewable and fail if the fixture stops reaching a
# real rank owner, nonblocking executor, collective poll, or both completion
# roles after release optimization.
for symbol in \
  'execute__rank__cycle(' \
  'execute__cycle(' \
  'finish__progress(' \
  'TensorParallelDeviceWorkerOwner12stage__frame(' \
  'TensorParallelDeviceWorkerOwner19progress__execution(' \
  'TensorParallelDeviceWorkerOwner21finish__leader__frame(' \
  'TensorParallelDeviceWorkerOwner16finish__follower(' \
  'TensorParallelDeviceWorkerOwner25graph__runtime__telemetry(' \
  'NcclCommunicator16poll__collective(' \
  'TensorCollectiveOwner4poll(' \
  'sample__leader__rows(' \
  'lunaflux_tp_alloc_evidence_reset' \
  'lunaflux_tp_alloc_collective_evidence_reset' \
  'lunaflux_device_worker_alloc_probe_begin'; do
  if ! rg -Fq "$symbol" "$generated_c"; then
    printf 'tensor-parallel allocation evidence symbol is missing: %s\n' \
      "$symbol" >&2
    exit 1
  fi
done

main_source="tests/tensor_parallel_device_worker_alloc/main.mbt"
warm_line=$(rg -n -m1 'execute_rank_cycle\(workers, cycles, 0\)' \
  "$main_source" | cut -d: -f1)
probe_line=$(rg -n 'alloc_probe_begin\(\)' "$main_source" | tail -n1 | \
  cut -d: -f1)
measured_line=$(rg -n -m1 \
  'execute_rank_cycle\(workers, cycles, index\)' "$main_source" | \
  cut -d: -f1)
if [ -z "$warm_line" ] || [ -z "$probe_line" ] || \
  [ -z "$measured_line" ] || [ "$warm_line" -ge "$probe_line" ] || \
  [ "$probe_line" -ge "$measured_line" ] || \
  ! rg -Fq 'alloc_probe_check(observed, 1)' "$main_source"; then
  printf '%s\n' \
    'tensor-parallel allocation preparation, warm, and measured order drifted' >&2
  exit 1
fi

if rg -n 'native-stub.*\.\.|"\.\./.*\.c"' \
  tests/tensor_parallel_device_worker_alloc/moon.pkg; then
  printf '%s\n' 'tensor-parallel allocation harness escaped its stub root' >&2
  exit 1
fi

for file in tests/tensor_parallel_device_worker_alloc/*; do
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    printf '%s exceeds the strict 499-line allocation harness budget\n' \
      "$file" >&2
    exit 1
  fi
done

echo "tensor-parallel device-worker warmed allocation evidence: ok"
