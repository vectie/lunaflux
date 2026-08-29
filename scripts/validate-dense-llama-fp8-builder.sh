#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

fail() {
  printf '%s\n' "dense-Llama FP8 builder boundary: $1" >&2
  exit 1
}

source_file=model/llama/fp8_w8a8.mbt
test_file=model/llama/fp8_w8a8_test.mbt
v2_test_file=model/llama/fp8_w8a8_v2_test.mbt
facade_dir=model/llama_fp8_weights

for v2_surface in \
  'pub fn inspect_full_context_file_v2(' \
  'pub fn inspect_paged_file_v2('; do
  rg -F -q "$v2_surface" "$facade_dir/inspect_v2.mbt" || {
    printf '%s\n' "dense Llama FP8 v2 inspection surface missing: $v2_surface" >&2
    exit 1
  }
done
api=model/llama/pkg.generated.mbti

for file in "$source_file" "$test_file" "$v2_test_file" "$facade_dir/inspect.mbt" \
  "$facade_dir/inspect_test.mbt" "$facade_dir/moon.pkg" "$api"; do
  [ -f "$file" ] || fail "missing required file: $file"
  case "$file" in
    *.mbt)
      lines=$(wc -l < "$file" | tr -d ' ')
      [ "$lines" -lt 500 ] || fail "$file exceeds the 499-line budget"
      ;;
  esac
done

rg -U -q '^#valtype\npub struct LlamaFp8W8A8Policy \{\n  priv version : Int\n\}' \
  "$source_file" || fail 'policy is not an opaque value type'
rg -U -q '^#valtype\npub struct LlamaFp8W8A8V2Policy \{\n  priv version : Int\n\}' \
  "$source_file" || fail 'v2 policy is not a separate opaque value type'
rg -F -x -q \
  'pub fn build_finite_e4m3_fp8_w8a8_v1(@spec.LlamaModelMetadata, LlamaFp8W8A8Policy) -> @plan.ModelPlan raise @plan.PlanValidationError' \
  "$api" || fail 'full-context builder API drifted'
rg -F -x -q \
  'pub fn build_paged_finite_e4m3_fp8_w8a8_v1(@spec.LlamaModelMetadata, LlamaFp8W8A8Policy, batch? : LlamaPagedBatchEnvelope) -> @plan.ModelPlan raise @plan.PlanValidationError' \
  "$api" || fail 'paged builder API drifted'
rg -F -x -q \
  'pub fn build_finite_e4m3_fp8_w8a8_v2(@spec.LlamaModelMetadata, LlamaFp8W8A8V2Policy) -> @plan.ModelPlan raise @plan.PlanValidationError' \
  "$api" || fail 'full-context v2 builder API drifted'
rg -F -x -q \
  'pub fn build_paged_finite_e4m3_fp8_w8a8_v2(@spec.LlamaModelMetadata, LlamaFp8W8A8V2Policy, batch? : LlamaPagedBatchEnvelope) -> @plan.ModelPlan raise @plan.PlanValidationError' \
  "$api" || fail 'paged v2 builder API drifted'

for anchor in \
  'let parameter_count = 3 + 9 * layer_count' \
  'let scale_count = 1 + 7 * layer_count' \
  'StorageDType::f8_e4m3_finite()' \
  'ScaleGranularity::per_tensor()' \
  'ActivationScalePolicy::dynamic_per_tensor_f32_v1()' \
  'ActivationScalePolicy::gated_mlp_external_and_post_silu_bound_expf_f32_v2()' \
  'Fp8DynamicActivationWorkspaceV2' \
  'AccumulatorDType::f32()' \
  'TiedFp8EmbeddingsUnsupported' \
  'build_with_execution('
do
  rg -F -q "$anchor" "$source_file" || fail "missing invariant: $anchor"
done

for evidence in \
  'assert_eq(operation.workspace_bytes(), 4L)' \
  'assert_eq(operation.workspace_bytes(), 8L)' \
  'plan validation rejects mixed v1 and v2 projection workspaces' \
  'content="83393dddd3800f45170b6a9343c1b830c0306e2466c6dfefb153ae3b46a8f49d"' \
  'content="5b595e6132e79caba501c0974ad5002749212324097292d09a66f440160dee48"'; do
  rg -F -q "$evidence" "$v2_test_file" || fail "missing v2 evidence: $evidence"
done

if rg -n 'scheduler|internal/cuda|@cuda\.|extern "c"|RuntimeDescriptor' \
  "$source_file" "$facade_dir" --glob '*.mbt' --glob moon.pkg; then
  fail 'builder or file facade gained runtime, CUDA, or readiness authority'
fi
if rg -n '^pub fn [A-Za-z0-9_:]*(execute|launch|load|ready|dispatch)\(|^pub (struct|enum) [A-Za-z0-9_]*Ready' \
  "$source_file" "$facade_dir" --glob '*.mbt'; then
  fail 'builder or file facade gained execution or readiness authority'
fi
if rg -n 'try![[:space:]]+@llama\.build_' "$facade_dir/inspect.mbt"; then
  fail 'public file facade aborts on plan-validation failure'
fi

for evidence in \
  'alias_name_ordinal=2' \
  'fp8_alias_ordinal=2' \
  'omit_ordinal=11' \
  'extra=true' \
  'overlap_ordinal=2' \
  'wrong_shape_ordinal=11' \
  'InvalidNumericWeightFile(NumericSchemaMismatch)' \
  'InvalidNumericWeightFile(ArtifactDigestMismatch)' \
  'paged FP8 inspection admits its exact plan and rejects identity replay' \
  'valid facade inputs preserve plan overflow as a typed rejection'
do
  rg -F -q "$evidence" "$facade_dir/inspect_test.mbt" || \
    fail "missing hostile file evidence: $evidence"
done

moon check model/llama --target native --deny-warn
moon test model/llama --target native --deny-warn
moon check model/llama_fp8_weights --target native --deny-warn
moon test model/llama_fp8_weights --target native --deny-warn
printf '%s\n' 'Dense-Llama FP8 builder boundary is valid.'
