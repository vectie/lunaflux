#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

fail() {
  echo "dense-Llama I8 builder boundary: $1" >&2
  exit 1
}

source_file="model/llama/i8_weight_only.mbt"
test_file="model/llama/i8_weight_only_test.mbt"
readme="model/llama/README.mbt.md"
api="model/llama/pkg.generated.mbti"
plan_limits="model/plan/limits.mbt"
plan_limit_test="model/plan/limits_wbtest.mbt"
spec_file="model/spec/llama.mbt"

for required_file in \
  "$source_file" "$test_file" "$readme" "$api" \
  "$plan_limits" "$plan_limit_test" "$spec_file"; do
  [ -f "$required_file" ] || fail "missing required file: $required_file"
done

source_policy_is_safe() {
  candidate=$1
  printf '%s\n' "$candidate" | \
    rg -U -q '^#valtype\npub struct LlamaI8WeightOnlyPolicy \{\n  priv version : Int\n\}'
}

if ! source_policy_is_safe '#valtype
pub struct LlamaI8WeightOnlyPolicy {
  priv version : Int
}'; then
  fail "source-policy positive control rejected the opaque value type"
fi
if source_policy_is_safe 'pub struct LlamaI8WeightOnlyPolicy(Int)'; then
  fail "source-policy positive control accepted a tuple"
fi
if source_policy_is_safe '#valtype
pub struct LlamaI8WeightOnlyPolicy {
  version : Int
}'; then
  fail "source-policy positive control accepted a public field"
fi
policy_source=$(sed -n \
  '/^#valtype$/,/^} derive(Debug, Eq)$/p' "$source_file")
source_policy_is_safe "$policy_source" || \
  fail "I8 policy source is not an opaque #valtype"

api_policy_is_safe() {
  candidate=$1
  printf '%s\n' "$candidate" | \
    rg -U -q '^pub struct LlamaI8WeightOnlyPolicy \{\n  // private fields\n\}'
}

if ! api_policy_is_safe 'pub struct LlamaI8WeightOnlyPolicy {
  // private fields
}'; then
  fail "API-opacity positive control rejected private fields"
fi
if api_policy_is_safe 'pub struct LlamaI8WeightOnlyPolicy(Int)'; then
  fail "API-opacity positive control accepted a tuple"
fi
if api_policy_is_safe 'pub struct LlamaI8WeightOnlyPolicy {
  version : Int
}'; then
  fail "API-opacity positive control accepted a public field"
fi
policy_api=$(sed -n \
  '/^pub struct LlamaI8WeightOnlyPolicy {$/,/^} derive/p' "$api")
api_policy_is_safe "$policy_api" || fail "generated I8 policy is forgeable"

expected_policy_methods=$(printf '%s\n' \
  'pub fn LlamaI8WeightOnlyPolicy::new(tied_embeddings~ : Bool) -> Self raise LlamaI8WeightOnlyPolicyError' \
  'pub fn LlamaI8WeightOnlyPolicy::untied_v1() -> Self' | sort)
observed_policy_methods=$(rg '^pub fn LlamaI8WeightOnlyPolicy::' "$api" | sort)
[ "$observed_policy_methods" = "$expected_policy_methods" ] || \
  fail "I8 policy acquired a partial-selection or bypass method"

for exact_api in \
  'pub fn build_symmetric_i8_weight_only_v1(@spec.LlamaModelMetadata, LlamaI8WeightOnlyPolicy) -> @plan.ModelPlan raise @plan.PlanValidationError' \
  'pub fn build_paged_symmetric_i8_weight_only_v1(@spec.LlamaModelMetadata, LlamaI8WeightOnlyPolicy, batch? : LlamaPagedBatchEnvelope) -> @plan.ModelPlan raise @plan.PlanValidationError'; do
  rg -F -x -q "$exact_api" "$api" || fail "exact builder API drifted: $exact_api"
done
[ "$(rg -c '^pub fn build_.*symmetric_i8_weight_only' "$api")" -eq 2 ] || \
  fail "I8 builder surface is not exactly full-context plus paged"
if rg -n '^pub fn .*symmetric_i8.*(layer|tensor|matrix|operation|select|filter|mask)' \
  "$api"; then
  fail "partial I8 selection became public"
fi

imports=$(rg -o '"vectie/lunaflux/[^"]+"' model/llama/moon.pkg | \
  tr -d '"' | sort)
expected_imports=$(printf '%s\n' \
  'vectie/lunaflux/model/numeric_contract' \
  'vectie/lunaflux/model/plan' \
  'vectie/lunaflux/model/spec' | sort)
[ "$imports" = "$expected_imports" ] || \
  fail "model-family builder import surface drifted"
if rg -n 'ModelIdentity|PlanDigest|scheduler|Scheduler|CUDA|NCCL|DeviceContext|KernelCatalog' \
  "$source_file"; then
  fail "builder acquired identity, scheduler, or backend authority"
fi

for invariant in \
  'let parameter_count = 3 + 9 * layer_count' \
  'let scale_count = 1 + 7 * layer_count' \
  'append_i8_llama_bf16_parameter(tensors, [vocabulary, hidden])' \
  'append_i8_llama_quantized_parameter(' \
  'append_i8_llama_scale(tensors, vocabulary)' \
  'operations.length() != 3 + 9 * layer_count' \
  'TensorStorageContract::symmetric_i8_weight_only_v1()' \
  'TensorStorageContract::symmetric_i8_weight_only_scale_v1()' \
  'OperationExecutionContract::symmetric_i8_weight_only_v1()' \
  'build_with_execution(' \
  'policy.validate()'; do
  rg -F -q "$invariant" "$source_file" || \
    fail "mandatory all-or-nothing source invariant is missing: $invariant"
done

[ "$(rg -F -c 'append_i8_llama_quantized_parameter(' "$source_file")" -eq 9 ] || \
  fail "eligible I8 parameter call sites no longer cover seven matrices per layer plus LM head"
[ "$(rg -F -c 'append_i8_llama_scale(' "$source_file")" -eq 9 ] || \
  fail "scale call sites no longer mirror every eligible I8 parameter"

policy_line=$(rg -n -F 'policy.validate()' "$source_file" | sed -n '1s/:.*//p')
schema_line=$(rg -n -F 'build_i8_llama_numeric_schema(metadata.spec())' \
  "$source_file" | sed -n '1s/:.*//p')
[ -n "$policy_line" ] && [ -n "$schema_line" ] && \
  [ "$policy_line" -lt "$schema_line" ] || \
  fail "policy admission no longer precedes builder arrays"
constructor=$(sed -n \
  '/^pub fn LlamaI8WeightOnlyPolicy::new(/,/^}$/p' "$source_file")
printf '%s\n' "$constructor" | rg -F -q 'if tied_embeddings {' || \
  fail "tied embeddings are no longer rejected"
if printf '%s\n' "$constructor" | rg -n 'Array|build_|ModelPlan'; then
  fail "tied-embedding rejection allocates or constructs a plan"
fi

for evidence in \
  'dense Llama I8 policy fails closed for tied embeddings' \
  'dense Llama I8 plan quantizes every and only eligible parameter' \
  'assert_false(scale_seen[scale_index])' \
  'scale_tensor.shape().dimension(0), tensor.shape().dimension(0)' \
  'dense Llama I8 operation contracts correspond exactly to graph kinds' \
  'dense Llama I8 identities are deterministic and mode-specific' \
  'content="c83f4ef9af74a24d710db0eb3ba6032b5750b7f8c52134afb755a84bba3e9e86"' \
  'content="386baef801ca914de2c27c2b82018f4694045837388d91fbcaa926ccf519ac4c"' \
  'maximum.operations().length(), 4089' \
  'maximum.numeric_binding().tensors().length(), 7268' \
  'total_value_inputs, 9086' \
  'layer_count=455'; do
  rg -F -q "$evidence" "$test_file" || \
    fail "builder evidence is missing: $evidence"
done

for contract_text in \
  'all-or-nothing' \
  'Q, K, V, output, gate, up, and down matrix' \
  'unique plain-F32 scale' \
  'Tied embeddings fail closed' \
  '4,089' \
  '7,268' \
  '9,086' \
  '16,384'; do
  rg -F -q "$contract_text" "$readme" || \
    fail "README contract is missing: $contract_text"
done

rg -F -q 'const LLAMA_MODEL_HARD_MAX_LAYER_COUNT : Int = 454' "$spec_file" || \
  fail "opaque model spec lost the pre-allocation layer ceiling"
for limit in \
  'model_plan_hard_max_operations : Int = 4096' \
  'model_plan_hard_max_total_value_inputs : Int = 16384' \
  'model_plan_hard_max_value_inputs_per_operation : Int = 64' \
  'model_plan_hard_max_runtime_inputs_per_operation : Int = 16' \
  'model_plan_hard_max_outputs_per_operation : Int = 1' \
  'model_plan_hard_max_canonical_bytes : Int = 16 * 1024 * 1024'; do
  rg -F -q "$limit" "$plan_limits" || fail "generic plan cap drifted: $limit"
done
rg -F -q 'limits_wb_invalid_kind(1, 16385, 1, 1, 1, 1)' \
  "$plan_limit_test" || fail "total-input hard-ceiling rejection evidence is missing"
[ $((6 + 20 * 454)) -eq 9086 ] || fail "value-input formula control failed"
[ 9086 -lt 16384 ] || fail "Llama maximum escaped the generic total-input cap"

for file in "$source_file" "$test_file" "$readme"; do
  lines=$(wc -l < "$file" | tr -d ' ')
  [ "$lines" -lt 500 ] || fail "$file exceeds the strict 499-line budget"
done

echo "dense-Llama symmetric-I8 builder boundary: ok"
