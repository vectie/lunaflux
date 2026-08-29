#!/bin/sh
set -eu

package_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
production=$(rg --files "$package_dir" | rg '/[^/]+\.mbt$' | rg -v 'test\.mbt$')

for file in $production; do
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    echo "tensor-parallel group transport production file is not focused: $file" >&2
    exit 1
  fi
done

if rg -n -i 'llama|dense_llama|nccl|cuda|internal/' $production; then
  echo 'tensor-parallel group transport leaks model-family or backend vocabulary' >&2
  exit 1
fi

if rg -n -i 'nccl|cuda|cublas' "$package_dir/types.mbt" "$package_dir/README.mbt.md"; then
  echo 'tensor-parallel group public vocabulary names a concrete backend' >&2
  exit 1
fi

if rg -n 'WorkerService|worker_service|scheduler/core|tensor_parallel_rank_child' $production; then
  echo 'tensor-parallel group transport crossed the service/scheduler/rank-child boundary' >&2
  exit 1
fi

if rg -n 'TensorParallelExecutionAdmission|KernelArtifactBundle|DeviceWeightLayout|DeviceWeightFileInspection' "$package_dir/types.mbt"; then
  echo 'group owner retains artifact bytes or a full-model weight representation' >&2
  exit 1
fi

if ! rg -Fq 'admitted_rank_child_activation_path' \
    "$package_dir/types.mbt" "$package_dir/prepare.mbt" ||
  ! rg -Fq '@worker_executable_file.WorkerExecutableAdmission' \
    "$package_dir/types.mbt" "$package_dir/prepare.mbt" ||
  rg -n '@worker_executable_file\.verify|ApprovedRoot::open_absolute' $production; then
  echo 'group executable input bypassed the deployment-owned admission boundary' >&2
  exit 1
fi

root_fields=$(rg -c 'priv roots : @approved_fs\.WorkerApprovedRoots' "$package_dir/types.mbt" || true)
if [ "$root_fields" -ne 1 ]; then
  echo 'group owner must retain exactly one approved worker-root pair' >&2
  exit 1
fi

for required in \
  'duplicate_worker_approved_root' \
  'reauthenticate(model_root)' \
  'create_group_id()' \
  'mint_group_generation' \
  'current_previous_plan_sequence_value' \
  'spawn_with_approved_root_spawn_authority' \
  'received.0.graph_telemetry(supervisor)' \
  'RankGroupProcessCleanupRequired'; do
  if ! rg -Fq "$required" $production; then
    echo "missing group-transport boundary evidence: $required" >&2
    exit 1
  fi
done

if ! grep -Fqx \
  'pub fn TensorParallelGroupTransport::graph_telemetry(Self, TensorParallelGroupReceivedCompletion) -> @worker_wire.WorkerGraphTelemetry raise TensorParallelGroupTransportError' \
  "$package_dir/pkg.generated.mbti" ||
  rg -n 'graph_(hits|misses).*\+' "$package_dir/process.mbt"; then
  echo 'group graph telemetry must forward one logical report without rank summation' >&2
  exit 1
fi

echo 'tensor-parallel group transport boundary checks passed'
