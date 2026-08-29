#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

require() {
  pattern=$1
  file=$2
  if ! rg -q "$pattern" "$root/$file"; then
    echo "tensor-parallel production boundary missing: $file / $pattern" >&2
    exit 1
  fi
}

require 'lunaflux\.launch\.v3' deploy/launch_file/schema.mbt
require 'DenseLlamaTensorParallelPagedAotApprovedV8' deploy/launch_file/types.mbt
require 'luna_approval is None' deploy/launch_file/schema.mbt
require 'approval\.manifest_digest\(\) != kernel\.manifest_digest\(\)' engine/worker_wire/bootstrap_source_tensor_parallel_v8.mbt
require 'approval\.approved_source\(\) != kernel\.manifest_digest\(\)' engine/worker_wire/bootstrap_source_tensor_parallel_v8.mbt
require 'has_valid_tensor_parallel_recipe_binding' engine/tensor_parallel_rank_configure/authenticate.mbt
require 'has_valid_tensor_parallel_recipe_binding' engine/tensor_parallel_rank_child/bootstrap_admission.mbt
require 'external_approval=source\.kernel\(\)\.luna_approval\(\)' engine/tensor_parallel_rank_child/bootstrap_admission.mbt
require 'admit_rank_ordered_local_topology' runtime/descriptor_file/tensor_parallel_load.mbt
require 'prepare_owned_tensor_parallel_luna_online_framed_service' ops/runtime_instance/runtime_route.mbt
require 'GroupReady' engine/worker_service/owned_prepare_tensor_parallel.mbt
require 'TensorParallelGroupTemplateDigest' engine/tensor_parallel_group_transport/types.mbt
require '!report\.has_canonical_failure_rank\(\)' engine/tensor_parallel_group_transport/failure_cleanup.mbt
require 'lunaflux\.tensor-parallel-group-template\.v1' engine/tensor_parallel_group_transport/template_digest.mbt
require 'admit_with_template' engine/tensor_parallel_group_transport/build.mbt
require 'lunaflux\.tensor-parallel-group-bootstrap\.v4' engine/tensor_parallel_worker_bootstrap/contract_digest.mbt
require 'bootstrap\.template_digest_sha256\(\)' engine/tensor_parallel_group_transport/build.mbt
require 'tensor_parallel_group_template_sha256=' ops/runtime_instance/release_preflight.mbt

if rg -q 'tensor_parallel_group_contract=deferred|0000000000000000000000000000000000000000000000000000000000000000' \
  "$root/ops/runtime_instance/release_preflight.mbt"; then
  echo "tensor-parallel preflight retained a deferred or zero group identity" >&2
  exit 1
fi

if rg -n 'physical NCCL (pass|validated)|physical_nccl_(pass|validated)' \
  "$root/engine" "$root/runtime" "$root/ops" "$root/deploy"; then
  echo "tensor-parallel production source makes a forbidden physical NCCL claim" >&2
  exit 1
fi

echo "tensor-parallel production route boundary: ok"
