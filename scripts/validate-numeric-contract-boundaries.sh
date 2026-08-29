#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

numeric_dir="model/numeric_contract"
plan_dir="model/plan"
for directory in "$numeric_dir" "$plan_dir"; do
  if [ ! -d "$directory" ]; then
    echo "numeric model package is missing: $directory" >&2
    exit 1
  fi
done

production_numeric_files=$(rg --files "$numeric_dir" \
  --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt')
production_plan_files=$(rg --files "$plan_dir" \
  --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt')

imports=$(sed -n '/^import {/,/^}/p' "$numeric_dir/moon.pkg" | \
  rg -v '^import \{|^}|moonbitlang/x/crypto|internal/canonical_sha256|^[[:space:]]*$' || true)
if [ -n "$imports" ]; then
  echo "numeric contract imports authority beyond hashing" >&2
  echo "$imports" >&2
  exit 1
fi

if rg -n 'TensorRef|vectie/lunaflux/|extern "c"|CUDA|NCCL|Device|Scheduler|KernelCatalog|safetensors|tokenizer|Llama' \
  "$numeric_dir" --glob '*.mbt' --glob '!**/*_test.mbt'; then
  echo "reference-free numeric contract acquired foreign authority" >&2
  exit 1
fi

for required in \
  'lunaflux.tensor-storage-contract.v1\x00' \
  'lunaflux.operation-execution-contract.v2\x00' \
  'lunaflux.model-numeric-schema.v2\x00' \
  'pub struct TensorStorageContract' \
  'pub struct OperationExecutionContract' \
  'pub struct ModelNumericSchema' \
  'pub struct NumericTensorSchema' \
  'pub struct NumericOperationSchema' \
  'pub fn load_tensor_storage_contract' \
  'pub fn load_operation_execution_contract' \
  'pub struct NumericShape' \
  'TensorStorageContract::symmetric_i8_weight_only_v1' \
  'OperationExecutionContract::symmetric_i8_weight_only_v1' \
  'symmetric_i8_weight_only_v1_accepts_code'; do
  if ! rg -F -q "$required" $production_numeric_files; then
    echo "numeric contract invariant is missing: $required" >&2
    exit 1
  fi
done

for required in \
  'closed interval `[-127, 127]`' \
  'Code `-128` is reserved' \
  '`scale = max_abs / 127.0`' \
  '`F32(code) * scale`'; do
  if ! rg -F -q "$required" "$numeric_dir/README.mbt.md"; then
    echo "symmetric-I8 exact semantic is missing: $required" >&2
    exit 1
  fi
done

if rg -n 'NumericContractDigest|pub struct NumericContract|WeightNumericPolicy|ActivationNumericPolicy|QuantizationPolicy|unquantized_bf16_v1|w8a8_v1|weight_only_per_output_channel_v1' \
  "$numeric_dir" --glob '*.mbt' --glob '*.mbt.md'; then
  echo "superseded model-wide numeric API remains reachable" >&2
  exit 1
fi

source_interface_is_safe() {
  candidate=$1
  printf '%s\n' "$candidate" | \
    rg -q '^pub struct [A-Za-z][A-Za-z0-9_]* \{$' || return 1
  printf '%s\n' "$candidate" | \
    rg -q '^  priv [a-z][A-Za-z0-9_]* :' || return 1
  if printf '%s\n' "$candidate" | \
    rg -q '^pub struct [A-Za-z][A-Za-z0-9_]*\(|^  (pub |pub\(all\) )?[a-z][A-Za-z0-9_]* :'; then
    return 1
  fi
  return 0
}

if ! source_interface_is_safe 'pub struct SourcePrivateControl {
  priv value : Int
} derive(Eq)'; then
  echo "source-opacity positive control rejected private fields" >&2
  exit 1
fi
if source_interface_is_safe 'pub struct SourceTupleControl(Int)'; then
  echo "source-opacity positive control accepted a tuple representation" >&2
  exit 1
fi
if source_interface_is_safe 'pub struct SourcePublicControl {
  value : Int
} derive(Eq)'; then
  echo "source-opacity positive control accepted a public field" >&2
  exit 1
fi

source_valtype_is_present() {
  candidate=$1
  printf '%s\n' "$candidate" | \
    rg -U -q '^#valtype\npub struct [A-Za-z][A-Za-z0-9_]* \{$'
}

if ! source_valtype_is_present '#valtype
pub struct SourceValtypeControl {'; then
  echo "source-valtype positive control rejected an adjacent annotation" >&2
  exit 1
fi
if source_valtype_is_present 'pub struct SourceReferenceControl {'; then
  echo "source-valtype positive control accepted a reference wrapper" >&2
  exit 1
fi

for type in \
  TensorStorageDigest OperationExecutionDigest ModelNumericSchemaDigest \
  NumericTensorOrdinal NumericOperationOrdinal ScaleTensorOrdinal \
  ZeroPointTensorOrdinal CodebookTensorOrdinal NumericTensorRole \
  ModelNumericSchemaLimits \
  StorageDType StorageLayout StorageEncoding ComputeDType ActivationScalePolicy ActivationScaleStage AccumulatorDType \
  RoundingPolicy SaturationPolicy FiniteValuePolicy OperationOrderPolicy \
  ScaleGranularity ZeroPointMode CodebookLayout; do
  declaration=$(rg --no-filename -U -o "#valtype\\npub struct ${type} \\{" \
    $production_numeric_files || true)
  if ! source_valtype_is_present "$declaration"; then
    echo "numeric source vocabulary lost #valtype representation: $type" >&2
    exit 1
  fi
done

for type in \
  TensorStorageDigest OperationExecutionDigest ModelNumericSchemaDigest \
  NumericTensorOrdinal NumericOperationOrdinal ScaleTensorOrdinal \
  ZeroPointTensorOrdinal CodebookTensorOrdinal NumericTensorRole \
  NumericConversion TensorStorageContract OperationExecutionContract \
  NumericTensorSchema NumericOperationSchema ModelNumericSchema NumericShape ModelNumericSchemaLimits \
  StorageDType StorageLayout StorageEncoding ComputeDType ActivationScalePolicy ActivationScaleStage AccumulatorDType \
  RoundingPolicy SaturationPolicy FiniteValuePolicy OperationOrderPolicy \
  ScaleGranularity ZeroPointMode CodebookLayout; do
  if rg -q "^pub struct ${type}\\(" $production_numeric_files; then
    echo "numeric source vocabulary has a tuple representation: $type" >&2
    exit 1
  fi
  source_interface=$(sed -n "/^pub struct ${type} {$/,/^}/p" \
    $production_numeric_files)
  if ! source_interface_is_safe "$source_interface"; then
    echo "numeric source vocabulary exposes fields: $type" >&2
    exit 1
  fi
done

for required in \
  'activation_compute~ : ComputeDType' \
  'activation_scale~ : ActivationScalePolicy' \
  'ActivationScalePolicy::absent_exact_v1' \
  'ActivationScalePolicy::dynamic_per_tensor_f32_v1' \
  'ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2' \
  'ActivationScalePolicy::gated_mlp_external_and_post_silu_f32_v1' \
  'ActivationScalePolicy::stage_count' \
  'ActivationScalePolicy::stage_at' \
  'ActivationScaleStage::external_operation_input_v1' \
  'ActivationScaleStage::post_silu_gate_up_product_v1'; do
  if ! rg -F -q "$required" $production_numeric_files; then
    echo "operation numeric v2 invariant is missing: $required" >&2
    exit 1
  fi
done

for required in \
  'pub struct ScaleTensorRef {' \
  'pub struct ZeroPointTensorRef {' \
  'pub struct CodebookTensorRef {' \
  'pub struct PlanTensor' \
  'pub struct OperationNumericBinding' \
  'pub struct ModelNumericBinding' \
  'fn valid_numeric_tensor_shape' \
  'fn validate_complete_numeric_tensor_use' \
  'UnusedParameterTensor' \
  'UnreferencedNumericMetadata' \
  'numeric_schema~ : @numeric_contract.ModelNumericSchema'; do
  if ! rg -F -q "$required" $production_plan_files; then
    echo "mandatory model numeric binding is missing: $required" >&2
    exit 1
  fi
done

numeric_api="$numeric_dir/pkg.generated.mbti"
plan_api="$plan_dir/pkg.generated.mbti"
if [ ! -f "$numeric_api" ]; then
  echo "generated numeric-contract interface is missing: $numeric_api" >&2
  exit 1
fi
if [ ! -f "$plan_api" ]; then
  echo "generated model-plan interface is missing: $plan_api" >&2
  exit 1
fi

opaque_interface_is_safe() {
  candidate=$1
  printf '%s\n' "$candidate" | \
    rg -q '^pub struct [A-Za-z][A-Za-z0-9_]* \{$' || return 1
  printf '%s\n' "$candidate" | rg -F -q '  // private fields' || return 1
  if printf '%s\n' "$candidate" | \
    rg -q '^pub struct [A-Za-z][A-Za-z0-9_]*\(|^  [a-z][A-Za-z0-9_]* :'; then
    return 1
  fi
  return 0
}

if ! opaque_interface_is_safe 'pub struct HostilePrivateControl {
  // private fields
} derive(Eq)'; then
  echo "opaque-interface positive control rejected private fields" >&2
  exit 1
fi
if opaque_interface_is_safe 'pub struct HostilePlan(Int)'; then
  echo "opaque-interface positive control accepted a tuple representation" >&2
  exit 1
fi
if opaque_interface_is_safe 'pub struct HostilePlan {
  value : Int
} derive(Eq)'; then
  echo "opaque-interface positive control accepted public fields" >&2
  exit 1
fi

for type in \
  TensorStorageDigest OperationExecutionDigest ModelNumericSchemaDigest \
  NumericTensorOrdinal NumericOperationOrdinal ScaleTensorOrdinal \
  ZeroPointTensorOrdinal CodebookTensorOrdinal NumericTensorRole \
  NumericConversion TensorStorageContract OperationExecutionContract \
  NumericTensorSchema NumericOperationSchema ModelNumericSchema NumericShape ModelNumericSchemaLimits \
  StorageDType StorageLayout StorageEncoding ComputeDType ActivationScalePolicy ActivationScaleStage AccumulatorDType \
  RoundingPolicy SaturationPolicy FiniteValuePolicy OperationOrderPolicy \
  ScaleGranularity ZeroPointMode CodebookLayout; do
  interface=$(sed -n "/^pub struct ${type} {$/,/^}/p" "$numeric_api")
  if ! opaque_interface_is_safe "$interface"; then
    echo "numeric-contract generated interface is not opaque: $type" >&2
    exit 1
  fi
  if rg -q "^pub struct ${type}\\(" "$numeric_api"; then
    echo "numeric-contract generated interface retains a tuple representation: $type" >&2
    exit 1
  fi
done

for type in \
  ModelPlan ModelPlanLimits PlanTensor OperationNumericBinding ModelNumericBinding \
  PlanOperation ShapeConstraints WorkspaceBounds KvCacheGeometry \
  OperationId TensorRef ActivationRef KvLayerId KernelCapabilityId \
  ScaleTensorRef ZeroPointTensorRef CodebookTensorRef; do
  interface=$(sed -n "/^pub struct ${type} {$/,/^}/p" "$plan_api")
  if ! opaque_interface_is_safe "$interface"; then
    echo "model-plan interface is not opaque: $type" >&2
    exit 1
  fi
  if rg -q "^pub struct ${type}\\(" "$plan_api"; then
    echo "model-plan interface retains a tuple representation: $type" >&2
    exit 1
  fi
done

if rg -n 'FixedArray\[Byte\]|Bytes::from_array|new_materializing|finish_bytes' \
  "$numeric_dir/canonical_writer.mbt" "$numeric_dir/canonical.mbt" \
  "$numeric_dir/schema_canonical.mbt"; then
  echo "numeric canonical materialization regained a proportional staging buffer" >&2
  exit 1
fi
for required in \
  'priv struct NumericCanonicalMaterializationCursor' \
  'Bytes::makei(expected, index => cursor.next(index, byte_at))' \
  'index != self.next_index' \
  'ModelNumericSchemaCanonicalCursor' \
  'materialize_numeric_canonical('; do
  if ! rg -F -q "$required" "$numeric_dir" --glob '*.mbt' \
    --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt'; then
    echo "one-buffer numeric canonical cursor invariant is missing: $required" >&2
    exit 1
  fi
done

for required in \
  'pub fn ModelPlan::new_stateless(content_digest~ : @spec.ContentDigest, limits~ : ModelPlanLimits, shape_constraints~ : ShapeConstraints, kv_cache_geometry~ : KvCacheGeometry, workspace_bounds~ : WorkspaceBounds, operations~ : ArrayView[PlanOperation], final_output~ : ActivationRef, numeric_schema~ : @numeric_contract.ModelNumericSchema) -> Self raise PlanValidationError' \
  'pub fn ModelPlan::new_with_kv_execution(content_digest~ : @spec.ContentDigest, limits~ : ModelPlanLimits, shape_constraints~ : ShapeConstraints, kv_cache_geometry~ : KvCacheGeometry, workspace_bounds~ : WorkspaceBounds, operations~ : ArrayView[PlanOperation], final_output~ : ActivationRef, kv_execution~ : ModelKvExecution, numeric_schema~ : @numeric_contract.ModelNumericSchema) -> Self raise PlanValidationError' \
  'pub fn ModelPlanLimits::new(max_operations~ : Int, max_numeric_tensors~ : Int, max_total_value_inputs~ : Int, max_value_inputs_per_operation~ : Int, max_runtime_inputs_per_operation~ : Int, max_outputs_per_operation~ : Int, max_canonical_bytes~ : Int) -> Self raise PlanValidationError' \
  'pub fn ModelPlanLimits::production() -> Self' \
  'pub fn ModelPlan::identity(Self) -> @spec.ModelIdentity' \
  'pub fn ModelPlan::shape_constraints(Self) -> ShapeConstraints' \
  'pub fn ModelPlan::kv_cache_geometry(Self) -> KvCacheGeometry' \
  'pub fn ModelPlan::workspace_bounds(Self) -> WorkspaceBounds' \
  'pub fn ModelPlan::operations(Self) -> ReadOnlyArray[PlanOperation]' \
  'pub fn ModelPlan::required_capabilities(Self) -> ReadOnlyArray[KernelCapabilityId]' \
  'pub fn ModelPlan::final_output(Self) -> ActivationRef' \
  'pub fn ModelPlan::kv_execution(Self) -> ModelKvExecution' \
  'pub fn ModelPlan::numeric_binding(Self) -> ModelNumericBinding' \
  'pub fn ModelPlan::tensor_count(Self) -> Int' \
  'pub fn ModelPlan::tensor(Self, TensorRef) -> PlanTensor?'; do
  if ! rg -F -q "$required" "$plan_api"; then
    echo "validated ModelPlan API changed: $required" >&2
    exit 1
  fi
done

if rg -n '^pub fn ModelPlan::new_(stateless|with_kv_execution).*(@spec\.ModelIdentity|@spec\.PlanDigest|identity~|plan_digest~)' \
  "$plan_api"; then
  echo "ModelPlan constructor accepts caller identity authority" >&2
  exit 1
fi

free_plan_mint='^pub fn .*-> .*\b(ModelPlan|ModelPlanLimits)\b'
if ! printf '%s\n' 'pub fn Forgery::from_raw() -> Array[ModelPlan]' | \
  rg -q "$free_plan_mint"; then
  echo "wrapped other-receiver model-plan mint positive control is ineffective" >&2
  exit 1
fi
if rg -n "$free_plan_mint" "$plan_api"; then
  echo "public function or foreign receiver can mint ModelPlan authority" >&2
  exit 1
fi

model_plan_limits_method_is_allowed() {
  case "$1" in
    new|production) return 0 ;;
    *) return 1 ;;
  esac
}

if model_plan_limits_method_is_allowed from_unchecked; then
  echo "ModelPlanLimits allowlist positive control accepted a bypass" >&2
  exit 1
fi
for method in $(rg -o '^pub fn ModelPlanLimits::[A-Za-z][A-Za-z0-9_]*' "$plan_api" | \
  sed 's/^pub fn ModelPlanLimits:://'); do
  if ! model_plan_limits_method_is_allowed "$method"; then
    echo "unexpected public ModelPlanLimits method: $method" >&2
    exit 1
  fi
done

model_plan_method_is_allowed() {
  case "$1" in
    final_output|identity|kv_cache_geometry|kv_execution|new_stateless|\
      new_with_kv_execution|numeric_binding|operations|required_capabilities|\
      shape_constraints|tensor|tensor_count|workspace_bounds) return 0 ;;
    *) return 1 ;;
  esac
}

if model_plan_method_is_allowed from_unchecked; then
  echo "ModelPlan API allowlist positive control accepted a bypass" >&2
  exit 1
fi
if ! model_plan_method_is_allowed new_stateless; then
  echo "ModelPlan API allowlist positive control rejected a constructor" >&2
  exit 1
fi
for method in $(rg -o '^pub fn ModelPlan::[A-Za-z][A-Za-z0-9_]*' "$plan_api" | \
  sed 's/^pub fn ModelPlan:://'); do
  if ! model_plan_method_is_allowed "$method"; then
    echo "unexpected public ModelPlan method: $method" >&2
    exit 1
  fi
done

package_minted_constructor='^pub fn (PlanTensor|OperationNumericBinding|ModelNumericBinding)::(new|from|wrap|revalidate)'
if ! printf '%s\n' 'pub fn PlanTensor::new() -> PlanTensor' | \
  rg -q "$package_minted_constructor"; then
  echo "package-minted constructor positive control is ineffective" >&2
  exit 1
fi
if rg -n "$package_minted_constructor" "$plan_api"; then
  echo "package-minted numeric view exposes a public constructor" >&2
  exit 1
fi

if rg -n 'numeric_schema\?|numeric_binding\?' $production_plan_files; then
  echo "ModelPlan numeric identity became optional" >&2
  exit 1
fi

if rg -n 'lunaflux\.model-numeric-schema|encode_model_numeric_schema' \
  $production_plan_files; then
  echo "model plan duplicated canonical numeric-schema authority" >&2
  exit 1
fi

crypto_files=$(rg -l '@crypto' $production_plan_files || true)
if [ -n "$crypto_files" ]; then
  echo "model-plan production hashing depends on external crypto authority" >&2
  echo "$crypto_files" >&2
  exit 1
fi

sha_file="internal/canonical_sha256/sha256.mbt"
if [ ! -f "$sha_file" ] || \
  ! rg -F -q 'pub struct CanonicalSha256 {' "$sha_file" || \
  ! rg -F -q 'ReadOnlyArray[UInt]' "$sha_file" || \
  ! rg -F -q 'block: FixedArray::make(64, 0)' "$sha_file" || \
  ! rg -F -q 'schedule: FixedArray::make(64, 0U)' "$sha_file"; then
  echo "private fixed-scratch model-plan SHA-256 authority is incomplete" >&2
  exit 1
fi
transform_body=$(sed -n '/fn CanonicalSha256::transform/,/^\/\/\/|$/p' "$sha_file")
if ! printf '%s\n' 'fn CanonicalSha256::transform() { FixedArray::make(64, 0) }' | \
  rg -q 'FixedArray::make|Array\[|Bytes::'; then
  echo "per-block allocation positive control is ineffective" >&2
  exit 1
fi
if printf '%s\n' "$transform_body" | \
  rg -n 'FixedArray::make|Array\[|Array::|Bytes::'; then
  echo "shared canonical SHA-256 transform allocates per block" >&2
  exit 1
fi

if ! rg -F -q 'lunaflux.model-plan.v1\x00' "$plan_dir/canonical_encoding.mbt" || \
  ! rg -F -q 'new_counting' "$plan_dir/canonical_encoding.mbt" || \
  ! rg -F -q 'new_hashing' "$plan_dir/canonical_encoding.mbt"; then
  echo "streaming model-plan canonical authority is incomplete" >&2
  exit 1
fi

if rg -n 'Array\[Byte\]|Bytes::from_array|fn canonical_bytes' \
  "$plan_dir/canonical_encoding.mbt"; then
  echo "model-plan encoder retained proportional canonical bytes" >&2
  exit 1
fi
for required in \
  'reinterpret_as_uint64()' \
  'self.reserve(bytes.length())' \
  'self.reserve(1)' \
  'self.reserve(8)' \
  'numeric_schema.operations().length() != operations.length()'; do
  if ! rg -F -q "$required" "$plan_dir/canonical_encoding.mbt"; then
    echo "model-plan encoder invariant is missing: $required" >&2
    exit 1
  fi
done
if rg -n 'produced\.contains|produced.*\.search' "$plan_dir" \
  --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt'; then
  echo "model-plan activation validation regained a linear scan" >&2
  exit 1
fi
numeric_count_line=$(rg -n 'numeric_schema\.operations\(\)\.length\(\) != operations\.length\(\)' \
  "$plan_dir/canonical_encoding.mbt" | sed 's/:.*//' | head -n 1)
map_line=$(rg -n 'let numeric_binding = map_numeric_schema' \
  "$plan_dir/construction.mbt" | \
  sed 's/:.*//' | head -n 1)
preflight_line=$(rg -n 'let canonical_bytes = preflight_model_plan' \
  "$plan_dir/construction.mbt" | \
  sed 's/:.*//' | head -n 1)
if [ -z "$numeric_count_line" ] || [ -z "$map_line" ] || \
  [ -z "$preflight_line" ] || [ "$preflight_line" -ge "$map_line" ]; then
  echo "numeric exact-count preflight no longer precedes schema mapping" >&2
  exit 1
fi

mint_pattern='@[A-Za-z_][A-Za-z0-9_]*\.(ModelIdentity::new|PlanDigest::from_sha256)'
if ! printf '%s\n' '@model_spec.ModelIdentity::new(content, plan)' | \
  rg -q "$mint_pattern"; then
  echo "model-plan mint positive control is ineffective" >&2
  exit 1
fi
mint_files=$(rg -l "$mint_pattern" $production_plan_files || true)
if [ "$mint_files" != "$plan_dir/canonical_encoding.mbt" ]; then
  echo "model-plan identity minting escaped canonical_encoding.mbt" >&2
  echo "$mint_files" >&2
  exit 1
fi

if rg -n 'Scheduler|worker_service|rank_group_process|CUDA|NCCL|Llama' \
  "$plan_dir/numeric_binding.mbt" "$plan_dir/plan.mbt"; then
  echo "generic model numeric binding acquired runtime or family policy" >&2
  exit 1
fi

llama_weights_api="model/llama_weights/pkg.generated.mbti"
materialize_api="model/materialize/pkg.generated.mbti"
for sealed in LlamaWeightBindings TensorBinding; do
  if ! rg -U -q "pub struct $sealed \\{[[:space:]]*// private fields[[:space:]]*\\}" \
    "$llama_weights_api"; then
    echo "llama weight evidence is externally constructible: $sealed" >&2
    exit 1
  fi
done
for sealed in MaterializationLimits TensorChecksum TensorChunk TensorReceipt \
  MaterializationReport HostTensor HostMaterialization; do
  if ! rg -U -q "pub struct $sealed \\{[[:space:]]*// private fields[[:space:]]*\\}" \
    "$materialize_api"; then
    echo "materialization evidence is externally constructible: $sealed" >&2
    exit 1
  fi
done
if ! rg -F -q \
  'pub fn bind(@spec.LlamaModelMetadata, @plan.ModelPlan, @safetensors.SafetensorsMetadata) -> LlamaWeightBindings' \
  "$llama_weights_api" || \
  ! rg -F -q \
  'pub fn MaterializationLimits::new(max_header_bytes~ : Int, max_tensor_count~ : Int, max_payload_bytes~ : Int64, max_host_bytes~ : Int64) -> Self raise MaterializationError' \
  "$materialize_api"; then
  echo "checked weight/materialization factory surface changed" >&2
  exit 1
fi
sealed_mint='^pub fn [a-z][A-Za-z0-9_]*\([^)]*\) -> .*\b(LlamaWeightBindings|TensorBinding|TensorChecksum|TensorChunk|TensorReceipt|MaterializationReport|HostTensor|HostMaterialization)\b'
if ! printf '%s\n' 'pub fn forge() -> Result[HostMaterialization, Error]' | \
  rg -q "$sealed_mint"; then
  echo "sealed evidence free-mint positive control is ineffective" >&2
  exit 1
fi
sealed_mints=$(rg -n "$sealed_mint" "$llama_weights_api" "$materialize_api" || true)
unexpected_sealed_mints=$(printf '%s\n' "$sealed_mints" | \
  rg -v 'pub fn bind\(|pub fn copy_to_host\(|pub fn visit\(' || true)
if [ -n "$unexpected_sealed_mints" ]; then
  echo "unexpected free mint for sealed weight/materialization evidence" >&2
  echo "$unexpected_sealed_mints" >&2
  exit 1
fi

constructor_files=$(rg -l '@plan\.ModelPlan::new_(stateless|with_kv_execution)' \
  --glob '*.mbt' . || true)
if ! printf '%s\n' '@spec.ModelIdentity::new(content, @spec.PlanDigest::from_sha256(hex))' | \
  rg -q 'ModelIdentity::new|PlanDigest::from_sha256'; then
  echo "constructor identity-reuse positive control is ineffective" >&2
  exit 1
fi
for file in $constructor_files; do
  constructor_count=$(rg -c '@plan\.ModelPlan::new_(stateless|with_kv_execution)' "$file")
  schema_count=$(rg -c 'numeric_schema[~=]' "$file" || true)
  content_count=$(rg -c 'content_digest[~=]' "$file" || true)
  limits_count=$(rg -c 'limits[~=]' "$file" || true)
  if [ "$schema_count" -lt "$constructor_count" ] || \
    [ "$content_count" -lt "$constructor_count" ] || \
    [ "$limits_count" -lt "$constructor_count" ]; then
    echo "ModelPlan construction lacks exact content, limits, or numeric schema: $file" >&2
    exit 1
  fi
  if rg -n 'ModelIdentity::new|PlanDigest::from_sha256' "$file"; then
    echo "ModelPlan constructor caller retains fabricated plan identity: $file" >&2
    exit 1
  fi
done

sh scripts/validate-model-spec-identity-boundary.sh

if ! rg -F -q 'let numeric_schema = spec.numeric_schema()' model/llama/builder.mbt || \
  ! rg -F -q 'numeric_schema~' model/llama/builder.mbt; then
  echo "Llama builder does not consume its identity-bound numeric schema" >&2
  exit 1
fi

for file in "$numeric_dir"/*.mbt "$plan_dir"/*.mbt; do
  case "$file" in
    *_test.mbt|*_wbtest.mbt) continue ;;
  esac
  lines=$(wc -l < "$file")
  if [ "$lines" -ge 500 ]; then
    echo "$file exceeds the strict 499-line production budget" >&2
    exit 1
  fi
done

# Keep this focused entry point authoritative for the generated-interface,
# value-representation, and constructor-scaling controls as well.
if ! scripts/validate-numeric-schema-scaling.sh; then
  exit 1
fi

echo "numeric contract and model binding boundaries: ok"
