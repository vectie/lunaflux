#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

package=engine/device_topology

if rg -n 'vectie/lunaflux/(scheduler|service|engine/worker|internal/(cuda|nccl))' \
  "$package/moon.pkg" ||
  rg -n 'extern "c"|@(core|worker_service|cuda|nccl)\.|raw_[A-Za-z0-9_]*\(' \
    "$package" --glob '*.mbt'; then
  printf '%s\n' 'device topology acquired execution or scheduler authority' >&2
  exit 1
fi

importers=$(rg -l 'vectie/lunaflux/engine/device_topology' . \
  --glob '**/moon.pkg' | sed 's#^\./##' | sort || true)
expected_importers='engine/device_step/moon.pkg
engine/execution_manifest_file/moon.pkg
engine/rank_group_process/moon.pkg
engine/tensor_parallel_device_plan/moon.pkg
engine/tensor_parallel_device_worker/moon.pkg
engine/tensor_parallel_execution_plan/moon.pkg
engine/tensor_parallel_group_transport/moon.pkg
engine/tensor_parallel_kv_plan/moon.pkg
engine/tensor_parallel_rank_child/moon.pkg
engine/tensor_parallel_rank_configure/moon.pkg
engine/tensor_parallel_worker_bootstrap/moon.pkg
ops/physical_topology_diagnostic/moon.pkg
ops/tensor_parallel_report/moon.pkg
runtime/descriptor_file/moon.pkg
runtime/tensor_parallel_admission/moon.pkg
tests/tensor_parallel_device_worker_alloc/moon.pkg'
if [ "$importers" != "$expected_importers" ]; then
  printf '%s\n%s\n' 'device topology importer allowlist drifted:' "$importers" >&2
  exit 1
fi

if ! rg -Fq 'Full mesh is LunaFlux' "$package/README.mbt.md" ||
  ! rg -Fq 'not a general NCCL requirement' "$package/README.mbt.md" ||
  ! rg -Fq 'HeterogeneousTarget' "$package/errors.mbt" ||
  ! rg -Fq 'declared.target != observed.target' "$package/admission.mbt" ||
  ! rg -Fq '@catalog.DeviceTarget::from_capability' \
    "$package/projection.mbt" ||
  ! rg -Fq 'LocalTopologyProbeProjection::from_local_inventory' \
    "$package/projection.mbt" ||
  ! rg -Fq '@device.can_access_peer(source, destination)' \
    "$package/projection.mbt" ||
  ! rg -Fq 'admit_local_topology_from_inventory' \
    "$package/admission.mbt" ||
  ! rg -Fq 'peer.destination_ordinal == destination' \
    "$package/admission.mbt"; then
  printf '%s\n' 'device topology fail-closed contract drifted' >&2
  exit 1
fi

if matches=$(sed -n \
  '/^pub fn LocalTopologyProbeProjection::from_local_inventory(/,/^}/p' \
  "$package/projection.mbt" | rg -n \
  'open_context|create_|enable_|@nccl|@cuda|worker|scheduler' 2>/dev/null); then
  printf '%s\n%s\n' \
    'physical topology projection acquired runtime authority:' "$matches" >&2
  exit 1
fi

while IFS= read -r file; do
  lines=$(wc -l < "$file" | tr -d ' ')
  if [ "$lines" -ge 500 ]; then
    printf 'device topology source exceeds file budget: %s (%s)\n' \
      "$file" "$lines" >&2
    exit 1
  fi
done < <(find "$package" -name '*.mbt' -type f | sort)

printf '%s\n' 'LunaFlux local device-topology boundary is valid.'
