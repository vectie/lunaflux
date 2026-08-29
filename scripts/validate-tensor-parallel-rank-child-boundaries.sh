#!/usr/bin/env bash
set -euo pipefail

package="engine/tensor_parallel_rank_child"
command_package="cmd/tensor_parallel_rank_child"

for required in \
  "$package/moon.pkg" \
  "$package/types.mbt" \
  "$package/configure.mbt" \
  "$package/readiness.mbt" \
  "$package/bootstrap_admission.mbt" \
  "$package/bootstrap_prepare.mbt" \
  "$package/control.mbt" \
  "$package/run.mbt" \
  "$command_package/moon.pkg" \
  "$command_package/main.mbt"; do
  if [ ! -f "$required" ]; then
    printf 'tensor-parallel rank-child file is missing: %s\n' "$required" >&2
    exit 1
  fi
done

if rg -n 'Scheduler|scheduler|model/llama|cuda|nccl|fixture|echo|fake' \
    "$package" --glob '*.mbt' >/dev/null; then
  printf '%s\n' 'rank child crossed scheduler, model-family, or backend policy' >&2
  exit 1
fi

for evidence in \
  TensorParallelRankConfigureFrameBuffer \
  copy_payload_to \
  contract.binding \
  poll_startup \
  readiness_contract \
  begin_ready \
  previous_sequence_value \
  open_configure_receiver \
  require_absolute_identity \
  admit_local_topology_from_inventory \
  admit_runtime_policy \
  inspect_sharded_file \
  load_tensor_parallel \
  admit_tensor_collective_runtime \
  tensor_parallel_device_worker.prepare \
  progress_control \
  abort_plan \
  begin_drain \
  close_worker_after_failure; do
  if ! rg -q "$evidence" "$package" --glob '*.mbt'; then
    printf 'rank-child production boundary is missing: %s\n' "$evidence" >&2
    exit 1
  fi
done

if ! rg -q '@child\.run\(\)' "$command_package/main.mbt" ||
   ! rg -q '@process\.exit_failure\(\)' "$command_package/main.mbt"; then
  printf '%s\n' 'rank child executable is not bound to the production owner loop' >&2
  exit 1
fi

if rg -n 'configure.*Bytes|payload.*Bytes|Ready\(.*fixture|echo' \
    "$package" --glob '*.mbt' >/dev/null; then
  printf '%s\n' 'rank child admits an opaque/fake Configure or readiness path' >&2
  exit 1
fi

for file in "$package"/*.mbt; do
  if [ "$(wc -l < "$file")" -ge 500 ]; then
    printf '%s exceeds the strict 499-line production budget\n' "$file" >&2
    exit 1
  fi
done

printf '%s\n' 'tensor-parallel rank-child static boundaries: ok'
