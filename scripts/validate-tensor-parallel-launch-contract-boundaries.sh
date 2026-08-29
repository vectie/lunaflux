#!/bin/sh
set -eu

package_dir="kernels/tensor_parallel_launch_contract"

if rg -n 'vectie/lunaflux/(engine|device$|internal/|scheduler|service)' \
  "$package_dir/moon.pkg"; then
  echo "tensor-parallel launch contract imports an owner/backend package" >&2
  exit 1
fi

if rg -n 'Device(Context|Allocation|Stream)|Communicator|Nccl|NCCL|cu[A-Z]|cuda|scheduler|worker|service|JIT|jit|compiler|module_load|function_load' \
  "$package_dir" --glob '*.mbt'; then
  echo "tensor-parallel launch contract leaks execution or selection authority" >&2
  exit 1
fi

if rg -n 'pub fn TensorParallel(AotLaunchContract(Set)?|Operand|RankConstants)::new' \
  "$package_dir" --glob '*.mbt'; then
  echo "tensor-parallel admitted output gained a public constructor" >&2
  exit 1
fi

if rg -n 'operands~|ArrayView\[.*Operand' "$package_dir/types.mbt" "$package_dir/admit.mbt"; then
  echo "caller can supply tensor-parallel operand geometry" >&2
  exit 1
fi

for required in \
  'validate_operation_plans(model_plan, rank_plan)' \
  'rank_plan.operation(operation.id())' \
  'geometry.kind() != operation.kind()' \
  'item.input_width != geometry.input_width()' \
  'item.output_width != geometry.output_width()' \
  'item.inner_width != geometry.inner_width()' \
  'item.query_head_count != geometry.query_head_count()' \
  'item.tokens_per_page != layout.tokens_per_page()'; do
  if ! rg -F "$required" "$package_dir" --glob '*.mbt' >/dev/null; then
    echo "artifact shard profile is not cross-authenticated: $required" >&2
    exit 1
  fi
done

for required in \
  'RankConstants(version=TP_RANK_CONSTANTS_VERSION)' \
  'TP_RANK_CONSTANTS_BYTES : Int64 = 32L' \
  'TP_RANK_CONSTANTS_ALIGNMENT : Int64 = 16L' \
  'TensorParallelRankConstants::encode_into'; do
  if ! rg -F "$required" "$package_dir" --glob '*.mbt' >/dev/null; then
    echo "rank-constants v2 ABI is missing: $required" >&2
    exit 1
  fi
done

for required in OperationCount OperationOrder OperationShape \
  FullTensorReplication FullKvReplication VendorImplementation ArtifactFamily; do
  if ! rg -n "$required" "$package_dir" --glob '*.mbt' >/dev/null; then
    echo "tensor-parallel rejection invariant is missing: $required" >&2
    exit 1
  fi
done

if rg -n -i 'llama|dense_llama|layer_count[[:space:]]*\*[[:space:]]*9|local_hidden|local_intermediate|local_vocabulary|local_query_heads|local_kv_heads|local_qkv_width|VocabularyLogits' \
    "$package_dir" \
    --glob '*.mbt' \
    --glob '!**/*_test.mbt' \
    --glob '!**/*_wbtest.mbt'; then
  echo "tensor-parallel launch bridge contains model-family debt" >&2
  exit 1
fi

for required in CollectiveSequence CollectiveOperation CollectiveWidth \
  LastAxisAllGather CollectiveLastAxisAllGather; do
  if ! rg -n "$required" "$package_dir/collectives.mbt" >/dev/null; then
    echo "exact collective validation is missing: $required" >&2
    exit 1
  fi
done

api="$package_dir/pkg.generated.mbti"
if [ ! -f "$api" ]; then
  echo "tensor-parallel launch generated API evidence is missing" >&2
  exit 1
fi
if rg -n 'TensorParallel(AotLaunchContract(Set)?|Operand|RankConstants)::new|DeviceContext|DeviceAllocation|Communicator|Scheduler' "$api"; then
  echo "tensor-parallel launch API leaks fabrication or owner authority" >&2
  exit 1
fi
if ! rg -n 'pub fn admit_paged_rank\(' "$api" >/dev/null ||
  ! rg -n 'pub fn TensorParallelAotLaunchContractInput::new\(' "$api" >/dev/null ||
  ! rg -n 'pub fn TensorParallelAotLaunchContract::operands\(' "$api" >/dev/null ||
  ! rg -n 'pub fn TensorParallelRankConstants::encode_into\(' "$api" >/dev/null; then
  echo "tensor-parallel launch API lost its focused admission/views" >&2
  exit 1
fi

for file in "$package_dir"/*.mbt; do
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    echo "$file exceeds the strict 499-line package budget" >&2
    exit 1
  fi
done

echo "tensor-parallel launch contract boundaries: ok"
