#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

require() {
  pattern=$1
  path=$2
  if ! rg -q -- "$pattern" "$path"; then
    echo "missing FP8 reusable production boundary: $pattern in $path" >&2
    exit 1
  fi
}

reject() {
  pattern=$1
  path=$2
  if rg -q -- "$pattern" "$path"; then
    echo "forbidden FP8 reusable production boundary: $pattern in $path" >&2
    exit 1
  fi
}

require 'load_paged_fp8_v5' engine/device_worker_bootstrap/prepare_fp8.mbt
require 'DenseLlamaFp8ReusablePagedAotApprovedV9' engine/device_worker_bootstrap/prepare.mbt
require 'admit_fp8_paged_graph_execution_blueprint_v3' engine/fp8_device_worker_v3/prepare.mbt
require 'prepare_fp8_paged_graph_executor_v3' engine/fp8_device_worker_v3/prepare.mbt
require 'release_authority\(\)\.digest' engine/fp8_device_worker_v3/admit.mbt
require 'execution\.compute_major\(\) == 12 && execution\.compute_minor\(\) == 0' \
  engine/device_worker_bootstrap/derive_fp8.mbt
require 'target\.compute_major\(\) == 12 && target\.compute_minor\(\) == 0' \
  engine/execution_manifest_file/fp8_v5_load.mbt
require 'target\.compute_major\(\) == 12 && target\.compute_minor\(\) == 0' \
  engine/fp8_device_executor/authority.mbt
require 'claims\.compute_major == 12 && claims\.compute_minor == 0' \
  runtime/descriptor_file/fp8_load.mbt
require 'prepare_owned_online_framed_v3' ops/runtime_instance/runtime_route.mbt
require 'load_fp8_reusable_v3' ops/runtime_instance/runtime_route.mbt
require 'load_fp8_reusable_v3_materialized' ops/runtime_instance/runtime_route.mbt
require 'load_mistral_v1_materialized' ops/runtime_instance/runtime_route.mbt
require 'load_tensor_parallel_materialized' ops/runtime_instance/runtime_route.mbt
require 'DenseLlamaFp8ReusablePagedAotApprovedV9 => fp8_loader\(\)' ops/runtime_instance/runtime_route.mbt
require 'source\.kernel\(\)\.luna_approval\(\) is None' runtime/fp8_release_worker/release_prepare.mbt
reject 'physical.*pass|readiness.*physical|physical.*validated' engine/fp8_device_worker_v3
reject '0000000000000000000000000000000000000000000000000000000000000000' ops/runtime_instance/release_preflight.mbt

moon check --target native --deny-warn --warn-list +73 engine/fp8_device_worker_v3
moon check --target native --deny-warn --warn-list +73 engine/device_worker_bootstrap
moon test --target native deploy/launch_file
moon test --target native engine/device_worker_bootstrap
moon test --target native engine/worker_wire/bootstrap_source_fp8_test.mbt
moon test --target native ops/runtime_instance/runtime_route_wbtest.mbt

echo "FP8 reusable production route boundary passed"
