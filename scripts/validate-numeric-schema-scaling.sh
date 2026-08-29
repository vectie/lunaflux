#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

fail() {
  echo "numeric schema scaling boundary: $1" >&2
  exit 1
}

numeric_dir="model/numeric_contract"
sha_dir="internal/canonical_sha256"
numeric_api="$numeric_dir/pkg.generated.mbti"
sha_api="$sha_dir/pkg.generated.mbti"
for path in "$numeric_dir" "$sha_dir" "$numeric_api" "$sha_api"; do
  [ -e "$path" ] || fail "missing $path"
done

production_numeric=$(rg --files "$numeric_dir" --glob '*.mbt' \
  --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt')

for control in \
  'priv canonical : Bytes' \
  'for previous_index in 0..<index' \
  '@crypto.sha256(canonical)' \
  'pub struct HostileTag(Int)' \
  'ModelNumericSchema::new(tensors=[], operations=[])'; do
  printf '%s\n' "$control" | rg -q \
    'canonical : Bytes|previous_index|@crypto|pub struct [A-Za-z]+\(|ModelNumericSchema::new\((?!limits=)' \
    --pcre2 || fail "positive control is ineffective: $control"
done

if rg -n 'priv canonical : Bytes' $production_numeric; then
  fail "numeric contracts retained proportional canonical bytes"
fi
if rg -n 'for previous_index|for previous.*tensors|\.contains\(.*(scale|zero|codebook)' \
  "$numeric_dir/schema_validation.mbt"; then
  fail "metadata ownership regained a nested or linear owner scan"
fi
if rg -n '@crypto|moonbitlang/x/crypto' $production_numeric "$sha_dir/sha256.mbt"; then
  fail "production numeric hashing escaped the shared fixed-scratch SHA"
fi
if rg -n 'PlanSha256|plan_sha256_round_constants' model/plan --glob '*.mbt'; then
  fail "model plan retained a duplicate SHA implementation"
fi

transform=$(sed -n '/fn CanonicalSha256::transform/,/^\/\/\/|$/p' \
  "$sha_dir/sha256.mbt")
if ! printf '%s\n' 'fn transform { FixedArray::make(64, 0) }' | \
  rg -q 'FixedArray::make|Array::|Bytes::'; then
  fail "transform-allocation positive control is ineffective"
fi
if printf '%s\n' "$transform" | rg -n 'FixedArray::make|Array::|Bytes::'; then
  fail "shared SHA transform allocates per block"
fi
for required in \
  'block: FixedArray::make(64, 0)' \
  'schedule: FixedArray::make(64, 0U)' \
  'priv mut finalized : Bool' \
  'priv mut cached_hex : String?' \
  'assert_false(hasher.update_byte(0x00))' \
  'Bytes::make(1024 * 1024, 0x61)'; do
  rg -F -q "$required" "$sha_dir" --glob '*.mbt' || \
    fail "shared SHA invariant is missing: $required"
done
if rg -n '^pub fn CanonicalSha256::(scratch|transformed|block|schedule)' "$sha_api"; then
  fail "shared SHA diagnostics leaked through its public API"
fi

for required in \
  'MODEL_NUMERIC_SCHEMA_HARD_MAX_TENSORS : Int = 65536' \
  'MODEL_NUMERIC_SCHEMA_HARD_MAX_OPERATIONS : Int = 4096' \
  'MODEL_NUMERIC_SCHEMA_HARD_MAX_RANK : Int = 16' \
  'MODEL_NUMERIC_SCHEMA_HARD_MAX_CANONICAL_BYTES : Int = 16 * 1024 * 1024' \
  'MODEL_NUMERIC_SCHEMA_CANONICAL_BASE : Int = 49' \
  'MODEL_NUMERIC_SCHEMA_TENSOR_BASE : Int = 101' \
  'MODEL_NUMERIC_SCHEMA_OPERATION_SIZE : Int = 67' \
  'TENSOR_STORAGE_CANONICAL_LENGTH : Int = 45' \
  'OPERATION_EXECUTION_CANONICAL_LENGTH : Int = 51' \
  'let scale_owners = FixedArray::make(tensors.length(), -1)' \
  'let zero_owners = FixedArray::make(tensors.length(), -1)' \
  'let codebook_owners = FixedArray::make(tensors.length(), -1)'; do
  rg -F -q "$required" "$numeric_dir" --glob '*.mbt' || \
    fail "bounded O(n) invariant is missing: $required"
done

preflight_line=$(rg -n 'let canonical_length = preflight_model_numeric_schema_size' \
  "$numeric_dir/schema_validation.mbt" | sed 's/:.*//')
owner_line=$(rg -n 'let scale_owners = FixedArray::make' \
  "$numeric_dir/schema_validation.mbt" | sed 's/:.*//')
copy_line=$(rg -n 'let owned_tensors = ReadOnlyArray::from_array' \
  "$numeric_dir/schema_validation.mbt" | sed 's/:.*//')
hash_line=$(rg -n 'let digest = digest_model_numeric_schema' \
  "$numeric_dir/schema_validation.mbt" | sed 's/:.*//')
[ "$preflight_line" -lt "$owner_line" ] && [ "$owner_line" -lt "$copy_line" ] && \
  [ "$copy_line" -lt "$hash_line" ] || fail "preflight no longer precedes owners/copies/hash"

missing_limits=$(rg -U -l --pcre2 \
  'ModelNumericSchema::new\(\r?\n(?![ \t]*limits=)' model kernels \
  --glob '*.mbt' --glob '!model/numeric_contract/schema.mbt' || true)
[ -z "$missing_limits" ] || {
  printf '%s\n' "$missing_limits" >&2
  fail "schema constructor call omits explicit checked limits"
}

opaque_interface_is_safe() {
  candidate=$1
  printf '%s\n' "$candidate" | rg -q '^pub struct [A-Za-z][A-Za-z0-9_]* \{$' || return 1
  printf '%s\n' "$candidate" | rg -F -q '  // private fields' || return 1
  printf '%s\n' "$candidate" | rg -q '^pub struct [A-Za-z][A-Za-z0-9_]*\(' && return 1
  return 0
}
if opaque_interface_is_safe 'pub struct HostileTag(Int)'; then
  fail "opaque-interface positive control accepted a tuple"
fi
if opaque_interface_is_safe 'pub struct HostileTag { value : Int }'; then
  fail "opaque-interface positive control accepted public fields"
fi

opaque_types='TensorStorageDigest OperationExecutionDigest ModelNumericSchemaDigest NumericTensorOrdinal NumericOperationOrdinal ScaleTensorOrdinal ZeroPointTensorOrdinal CodebookTensorOrdinal NumericTensorRole StorageDType StorageLayout StorageEncoding ComputeDType ActivationScalePolicy ActivationScaleStage AccumulatorDType RoundingPolicy SaturationPolicy FiniteValuePolicy OperationOrderPolicy ScaleGranularity ZeroPointMode CodebookLayout ModelNumericSchemaLimits NumericConversion NumericShape TensorStorageContract OperationExecutionContract NumericTensorSchema NumericOperationSchema ModelNumericSchema'
for type in $opaque_types; do
  interface=$(sed -n "/^pub struct ${type} {$/,/^}/p" "$numeric_api")
  opaque_interface_is_safe "$interface" || \
    fail "generated API exposes numeric authority: $type"
done

is_source_valtype() {
  candidate=$1
  printf '%s\n' "$candidate" | rg -U -q '^#valtype\npub struct [A-Za-z][A-Za-z0-9_]* \{'
}
if is_source_valtype 'pub struct HostileTag { priv value : Int }'; then
  fail "valtype positive control accepted a heap record"
fi
valtype_wrappers='TensorStorageDigest OperationExecutionDigest ModelNumericSchemaDigest NumericTensorOrdinal NumericOperationOrdinal ScaleTensorOrdinal ZeroPointTensorOrdinal CodebookTensorOrdinal NumericTensorRole StorageDType StorageLayout StorageEncoding ComputeDType ActivationScalePolicy ActivationScaleStage AccumulatorDType RoundingPolicy SaturationPolicy FiniteValuePolicy OperationOrderPolicy ScaleGranularity ZeroPointMode CodebookLayout ModelNumericSchemaLimits'
for type in $valtype_wrappers; do
  declaration=$(rg --no-filename -U -o "#valtype\npub struct ${type} \\{" "$numeric_dir" \
    --glob '*.mbt' --glob '!**/*test.mbt' || true)
  is_source_valtype "$declaration" || \
    fail "numeric scalar wrapper lost #valtype: $type"
done
if rg -n '^pub type|^pub\(all\) type' "$numeric_api"; then
  fail "numeric authority can escape through a public alias"
fi

self_factories() {
  type=$1
  rg "^pub fn ${type}::[A-Za-z][A-Za-z0-9_]*.* -> Self" "$numeric_api" | \
    sed "s/^pub fn ${type}:://; s/(.*//" | sort | tr '\n' ' ' | sed 's/ $//'
}
assert_self_factories() {
  type=$1
  expected=$2
  actual=$(self_factories "$type")
  [ "$actual" = "$expected" ] || {
    printf '%s\n' "$actual" >&2
    fail "exact self-factory allowlist drifted for $type"
  }
}
if [ "$(printf '%s\n' \
  'pub fn Hostile::wrap() -> Option[ModelNumericSchemaDigest]' | \
  rg -c --pcre2 '^pub fn .* -> .*(Array|Result|Option).*(ModelNumericSchemaDigest)')" -ne 1 ] || \
  [ "$(printf '%s\n' 'pub fn StorageDType::from_raw(Int) -> Self' | \
  rg -c '^pub fn StorageDType::[A-Za-z].* -> Self')" -ne 1 ]; then
  fail "factory/container-mint positive controls are ineffective"
fi
assert_self_factories AccumulatorDType 'f32 i32'
assert_self_factories ActivationScalePolicy 'absent_exact_v1 dynamic_per_tensor_f32_v1 gated_mlp_external_and_post_silu_bound_expf_f32_v2 gated_mlp_external_and_post_silu_f32_v1'
assert_self_factories ActivationScaleStage 'external_operation_input_v1 post_silu_gate_up_product_v1'
assert_self_factories CodebookLayout 'absent contiguous_f32_values_v1'
assert_self_factories CodebookTensorOrdinal 'new'
assert_self_factories ComputeDType 'absent bfloat16 f32 f8_e4m3_finite i8'
assert_self_factories FiniteValuePolicy 'reject_non_finite_v1'
assert_self_factories ModelNumericSchema 'new'
assert_self_factories ModelNumericSchemaDigest 'from_sha256'
assert_self_factories ModelNumericSchemaLimits 'new production'
assert_self_factories NumericConversion 'exact_v1 new quantized_v1'
assert_self_factories NumericOperationOrdinal 'new'
assert_self_factories NumericOperationSchema 'new'
assert_self_factories NumericShape 'new'
assert_self_factories NumericTensorOrdinal 'new'
assert_self_factories NumericTensorRole 'codebook_metadata parameter scale_metadata zero_point_metadata'
assert_self_factories NumericTensorSchema 'new'
assert_self_factories OperationExecutionContract 'new symmetric_i8_weight_only_v1'
assert_self_factories OperationExecutionDigest 'from_sha256'
assert_self_factories OperationOrderPolicy 'strict_declared_order_v1'
assert_self_factories RoundingPolicy 'exact_no_rounding_v1 nearest_ties_to_even_v1'
assert_self_factories SaturationPolicy 'finite_storage_range_v1 none_v1'
assert_self_factories ScaleGranularity 'absent per_output_channel per_tensor'
assert_self_factories ScaleTensorOrdinal 'new'
assert_self_factories StorageDType 'bfloat16 f32 f8_e4m3_finite i8'
assert_self_factories StorageEncoding 'affine_v1 codebook_v1 plain_v1 scaled_v1 symmetric_v1'
assert_self_factories StorageLayout 'dense_row_major_v1'
assert_self_factories TensorStorageContract 'new symmetric_i8_weight_only_scale_v1 symmetric_i8_weight_only_v1'
assert_self_factories TensorStorageDigest 'from_sha256'
assert_self_factories ZeroPointMode 'absent implicit_symmetric_zero_v1 per_output_channel_tensor_v1'
assert_self_factories ZeroPointTensorOrdinal 'new'

wrapped_self=$(rg '^pub fn .* -> .*Self' "$numeric_api" | \
  rg -v -- ' -> Self( raise [A-Za-z][A-Za-z0-9_]*)?$' || true)
[ -z "$wrapped_self" ] || fail "wrapped Self mint escaped the exact factories"
authority_pattern='AccumulatorDType|ActivationScalePolicy|ActivationScaleStage|CodebookLayout|CodebookTensorOrdinal|ComputeDType|FiniteValuePolicy|ModelNumericSchema|ModelNumericSchemaDigest|ModelNumericSchemaLimits|NumericConversion|NumericOperationOrdinal|NumericOperationSchema|NumericShape|NumericTensorOrdinal|NumericTensorRole|NumericTensorSchema|OperationExecutionContract|OperationExecutionDigest|OperationOrderPolicy|RoundingPolicy|SaturationPolicy|ScaleGranularity|ScaleTensorOrdinal|StorageDType|StorageEncoding|StorageLayout|TensorStorageContract|TensorStorageDigest|ZeroPointMode|ZeroPointTensorOrdinal'
authority_returns=$(rg "^pub fn .* -> .*(${authority_pattern})" "$numeric_api" | \
  sed 's/(.*//' | sort)
expected_authority_returns=$(printf '%s\n' \
  'pub fn ActivationScalePolicy::stage_at' \
  'pub fn CodebookTensorOrdinal::tensor' \
  'pub fn ModelNumericSchema::digest' \
  'pub fn ModelNumericSchema::operations' \
  'pub fn ModelNumericSchema::tensors' \
  'pub fn NumericConversion::finite_values' \
  'pub fn NumericConversion::rounding' \
  'pub fn NumericConversion::saturation' \
  'pub fn NumericOperationSchema::execution' \
  'pub fn NumericOperationSchema::ordinal' \
  'pub fn NumericTensorSchema::codebook' \
  'pub fn NumericTensorSchema::ordinal' \
  'pub fn NumericTensorSchema::role' \
  'pub fn NumericTensorSchema::scale' \
  'pub fn NumericTensorSchema::shape' \
  'pub fn NumericTensorSchema::storage' \
  'pub fn NumericTensorSchema::zero_point' \
  'pub fn OperationExecutionContract::accumulator' \
  'pub fn OperationExecutionContract::activation_compute' \
  'pub fn OperationExecutionContract::activation_input' \
  'pub fn OperationExecutionContract::activation_scale' \
  'pub fn OperationExecutionContract::conversion' \
  'pub fn OperationExecutionContract::digest' \
  'pub fn OperationExecutionContract::order' \
  'pub fn OperationExecutionContract::output' \
  'pub fn OperationExecutionContract::tensor_input' \
  'pub fn ScaleTensorOrdinal::tensor' \
  'pub fn TensorStorageContract::codebook_layout' \
  'pub fn TensorStorageContract::conversion' \
  'pub fn TensorStorageContract::digest' \
  'pub fn TensorStorageContract::dtype' \
  'pub fn TensorStorageContract::encoding' \
  'pub fn TensorStorageContract::layout' \
  'pub fn TensorStorageContract::scale_granularity' \
  'pub fn TensorStorageContract::zero_point' \
  'pub fn ZeroPointTensorOrdinal::tensor' \
  'pub fn load_operation_execution_contract' \
  'pub fn load_tensor_storage_contract' | sort)
[ "$authority_returns" = "$expected_authority_returns" ] || \
  fail "direct or wrapped numeric authority return allowlist drifted"

free_values=$(sed -n '/^\/\/ Values$/,/^\/\/ Errors$/p' "$numeric_api" | \
  rg '^pub fn ' | sort)
expected_free_values=$(printf '%s\n' \
  'pub fn load_operation_execution_contract(Bytes, OperationExecutionDigest) -> OperationExecutionContract raise NumericContractError' \
  'pub fn load_tensor_storage_contract(Bytes, TensorStorageDigest) -> TensorStorageContract raise NumericContractError' \
  'pub fn storage_byte_length(StorageDType, NumericShape) -> Int64 raise NumericContractError' \
  'pub fn symmetric_i8_weight_only_v1_accepts_code(Int) -> Bool' | sort)
[ "$free_values" = "$expected_free_values" ] || fail "free numeric mint allowlist drifted"

for required in \
  'maximum tensor table remains one bounded linear construction' \
  'scale zero and codebook metadata each have one exact owner' \
  'canonical overflow rejects before owner tables copies and hash' \
  'shape rejects rank above the global hard cap before owning dimensions'; do
  rg -F -q "$required" "$numeric_dir" --glob '*test.mbt' || \
    fail "scaling evidence is missing: $required"
done

for required in \
  'const LLAMA_MODEL_HARD_MAX_LAYER_COUNT : Int = 454' \
  '3 + 9 * maximum.layer_count(), 4089' \
  'layer_count=455' \
  'num_hidden_layers\":455'; do
  rg -F -q "$required" model/spec model/config_reader --glob '*.mbt' || \
    fail "Llama pre-allocation ceiling is missing: $required"
done

for directory in "$numeric_dir" "$sha_dir"; do
  for source in "$directory"/*.mbt; do
    lines=$(wc -l < "$source" | tr -d ' ')
    [ "$lines" -lt 500 ] || fail "$source exceeds the 499-line source budget"
  done
done

echo "numeric schema scaling boundary: ok"
