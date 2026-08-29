#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
package="$root/engine/execution_manifest_file"
artifact="$root/kernels/artifact_file"

fail() {
  echo "tensor-parallel manifest boundary: $1" >&2
  exit 1
}

for file in \
  "$package/tensor_parallel_types.mbt" \
  "$package/tensor_parallel_derive.mbt" \
  "$package/tensor_parallel_load.mbt" \
  "$package/tensor_parallel_fixture_wbtest.mbt" \
  "$package/tensor_parallel_load_wbtest.mbt"
do
  lines=$(wc -l < "$file")
  [ "$lines" -lt 500 ] || fail "$(basename "$file") exceeds 499 lines"
done

production=$(sed -n '1,/supported_targets/p' "$package/moon.pkg")
printf '%s\n' "$production" | grep -Eq \
  'model/llama|model/llama_tensor_parallel|internal/(cuda|nccl)|rank_group|worker_service' &&
  fail "production package imports family, backend, rank-wire, or service authority"

tp_sources="$package/tensor_parallel_types.mbt $package/tensor_parallel_derive.mbt $package/tensor_parallel_load.mbt"
grep -En \
  'llama|DenseLlama|DeviceWeightLayout|plan_layout|open_(context|device)|@cuda|@nccl|rank_group|worker_service' \
  $tp_sources && fail "TP admission contains family/backend/full-weight authority"

grep -q 'read_manifest_snapshot' "$package/tensor_parallel_load.mbt" ||
  fail "TP loader bypasses canonical snapshot reader"
grep -q 'parse_manifest' "$package/tensor_parallel_load.mbt" ||
  fail "TP loader bypasses canonical manifest parser"
grep -q 'derive_catalog' "$package/tensor_parallel_load.mbt" ||
  fail "TP loader bypasses semantic catalog derivation"
grep -q 'derive_artifact_source' "$package/tensor_parallel_load.mbt" ||
  fail "TP loader bypasses canonical artifact claims"
grep -q 'admit_paged_rank' "$package/tensor_parallel_derive.mbt" ||
  fail "TP loader does not admit exact rank-local contracts"
grep -q 'load_tensor_parallel_admitted' "$package/tensor_parallel_load.mbt" ||
  fail "TP loader bypasses exact artifact-file admission"
grep -q 'tensor_parallel_execution_plan.admit' "$package/tensor_parallel_load.mbt" ||
  fail "TP loader does not produce the physical execution plan"
grep -q 'TensorParallelSpecialization' "$package/tensor_parallel_load.mbt" ||
  fail "TP loader does not reject unsupported v3 specialization"

grep -q 'load_admitted_inputs' "$artifact/load.mbt" ||
  fail "artifact extension bypasses canonical bounded module loading"
grep -q 'artifact.admit_tensor_parallel' "$artifact/load.mbt" ||
  fail "artifact extension bypasses tensor-parallel authentication"

echo "tensor-parallel manifest admission boundary: ok"
