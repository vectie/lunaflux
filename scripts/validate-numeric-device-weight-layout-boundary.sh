#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

materialize_dir="model/device_materialize"
device_plan_dir="engine/device_plan"
profile_dir="engine/device_profile"

fail() {
  echo "numeric device-weight layout boundary: $1" >&2
  exit 1
}

for directory in "$materialize_dir" "$device_plan_dir" "$profile_dir"; do
  [ -d "$directory" ] || fail "missing package: $directory"
done

materialize_production=$(rg --files "$materialize_dir" --glob '*.mbt' \
  --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt')
device_plan_production=$(rg --files "$device_plan_dir" --glob '*.mbt' \
  --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt')
profile_production=$(rg --files "$profile_dir" --glob '*.mbt' \
  --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt')
[ -n "$materialize_production" ] || fail "materialize production discovery is empty"
[ -n "$device_plan_production" ] || fail "device-plan production discovery is empty"
[ -n "$profile_production" ] || fail "device-profile production discovery is empty"

production_has() {
  pattern=$1
  shift
  rg -F -q "$pattern" "$@"
}

for required in \
  'pub fn plan_numeric_layout(' \
  'const MAX_NUMERIC_LAYOUT_TENSORS : Int = 65536' \
  '@numeric_contract.storage_byte_length(' \
  'numeric_schema_digest: Some(model_plan.numeric_binding().digest())' \
  'let next_end = offset + byte_length' \
  'let next_materialized = materialized + byte_length' \
  'numeric_schema_digest: None'; do
  production_has "$required" $materialize_production || \
    fail "missing numeric-layout invariant: $required"
done

if rg -n '\.byte_length\(\)|@device|@llama_weights|@materialize|@safetensors|@approved_fs|extern "c"' \
  "$materialize_dir/numeric_layout.mbt"; then
  fail "pure numeric layout acquired legacy-size, file, or device authority"
fi

for required in \
  'validate_graph_bounds(operations)' \
  'validate_weight_layout(model_plan, weight_layout)' \
  'model_plan.tensor_count()' \
  'numeric_tensor_byte_length(model_plan, expected_reference)' \
  'region.offset_bytes != expected_offset' \
  'materialized != materialized_bytes' \
  'previous_end != arena_bytes' \
  'weight_layout.numeric_schema_digest() !=' \
  'Some(model_plan.numeric_binding().digest())' \
  'binding.operation_execution_digest() != Some(execution_digest)' \
  'binding.aot_entry_point() is None' \
  'binding.semantic_version().as_int() != catalog_version.as_int()' \
  'operation_execution_digest: execution_digest' \
  'numeric_schema_digest: numeric_binding.digest()' \
  'weight_layout,'; do
  production_has "$required" $device_plan_production || \
    fail "missing static numeric-layout join: $required"
done

if rg -n 'expected_bf16_weight_bytes|Llama|safetensors|launch_contract|artifact|internal/(cuda|nccl)|extern "c"' \
  $device_plan_production; then
  fail "device plan retained model-kind inference or acquired execution authority"
fi

allowed_device_plan_import() {
  case "$1" in
    vectie/lunaflux/kernels/catalog|\
      vectie/lunaflux/model/device_materialize|\
      vectie/lunaflux/model/numeric_contract|\
      vectie/lunaflux/model/plan|\
      vectie/lunaflux/model/spec) return 0 ;;
    *) return 1 ;;
  esac
}

if allowed_device_plan_import vectie/lunaflux/kernels/artifact; then
  fail "device-plan import allowlist positive control accepted artifact authority"
fi
if ! allowed_device_plan_import vectie/lunaflux/model/numeric_contract; then
  fail "device-plan import allowlist positive control rejected numeric authority"
fi
if printf '%s\n' \
  'import {' \
  '  "production-authority",' \
  '}' \
  'import {' \
  '  "test-only-authority",' \
  '} for "test"' | sed -n '1,/^}/p' | rg -q 'test-only-authority'; then
  fail "production-import parser positive control included a test import"
fi
if ! printf '%s\n' \
  'import {' \
  '  "production-authority",' \
  '}' \
  'import {' \
  '  "test-only-authority",' \
  '} for "test"' | sed -n '1,/^}/p' | rg -q 'production-authority'; then
  fail "production-import parser positive control omitted a production import"
fi
device_plan_imports=$(sed -n '1,/^}/p' "$device_plan_dir/moon.pkg" | \
  sed -n 's/^[[:space:]]*"\([^"]*\)".*/\1/p')
for imported in $device_plan_imports; do
  allowed_device_plan_import "$imported" || \
    fail "device plan has an unauthorized production import: $imported"
done

for required in \
  'static_plan.catalog_version() == @catalog.CatalogVersion::v4()' \
  'raise InvalidPlan(CatalogVersion)'; do
  production_has "$required" "$profile_dir/planner.mbt" || \
    fail "legacy profile does not fail closed for v4: $required"
done

opaque_interface_is_safe() {
  candidate=$1
  printf '%s\n' "$candidate" | \
    rg -q '^pub struct [A-Za-z][A-Za-z0-9_]* \{$' || return 1
  printf '%s\n' "$candidate" | rg -F -q '  // private fields' || return 1
  if printf '%s\n' "$candidate" | \
    rg -q '^pub struct [A-Za-z][A-Za-z0-9_]*\('; then
    return 1
  fi
  return 0
}

if opaque_interface_is_safe 'pub struct HostileLayout(Int)'; then
  fail "opacity positive control accepted a tuple representation"
fi
if opaque_interface_is_safe 'pub struct HostileLayout {'; then
  fail "opacity positive control accepted exposed fields"
fi

for specification in \
  "$materialize_dir/pkg.generated.mbti:DeviceWeightRegion" \
  "$materialize_dir/pkg.generated.mbti:DeviceWeightLayout" \
  "$materialize_dir/pkg.generated.mbti:DeviceWeights" \
  "$materialize_dir/pkg.generated.mbti:DeviceWeightFileLimits" \
  "$device_plan_dir/pkg.generated.mbti:StaticOperation" \
  "$device_plan_dir/pkg.generated.mbti:StaticDevicePlan" \
  "$profile_dir/pkg.generated.mbti:ExactExecutionShape" \
  "$profile_dir/pkg.generated.mbti:ExactKernelShape" \
  "$profile_dir/pkg.generated.mbti:ProfileRegionId" \
  "$profile_dir/pkg.generated.mbti:ExecutionRegionView" \
  "$profile_dir/pkg.generated.mbti:ActivationValueView" \
  "$profile_dir/pkg.generated.mbti:OperationExecutionProfile" \
  "$profile_dir/pkg.generated.mbti:FinalRowView" \
  "$profile_dir/pkg.generated.mbti:FinalOutputView" \
  "$profile_dir/pkg.generated.mbti:DeviceExecutionProfile"; do
  interface_file=${specification%%:*}
  type=${specification##*:}
  interface=$(sed -n "/^pub struct ${type} {$/,/^} derive\|^}$/p" \
    "$interface_file")
  opaque_interface_is_safe "$interface" || \
    fail "validated generated interface is not opaque: $type"
done

for required_profile_api in \
  'pub fn ExactExecutionShape::full_prefill(batch_rows~ : Int, sequence_tokens~ : Int) -> Self raise DeviceProfileError' \
  'pub fn ExactExecutionShape::full_recompute(batch_rows~ : Int, sequence_tokens~ : Int) -> Self raise DeviceProfileError' \
  'pub fn ProfileRegionId::from_stable_int(Int) -> Self raise DeviceProfileError' \
  'pub fn DeviceExecutionProfile::operation(Self, @plan.OperationId) -> OperationExecutionProfile?' \
  'pub fn DeviceExecutionProfile::final_output(Self) -> FinalOutputView'; do
  rg -F -q "$required_profile_api" "$profile_dir/pkg.generated.mbti" || \
    fail "generated checked profile API drifted: $required_profile_api"
done

for required_api in \
  'pub fn plan_numeric_layout(@plan.ModelPlan, alignment_bytes~ : Int64, max_arena_bytes~ : Int64) -> DeviceWeightLayout raise DeviceWeightError' \
  'pub fn DeviceWeightLayout::numeric_schema_digest(Self) -> @numeric_contract.ModelNumericSchemaDigest?' \
  'pub fn StaticDevicePlan::numeric_schema_digest(Self) -> @numeric_contract.ModelNumericSchemaDigest' \
  'pub fn StaticDevicePlan::weight_layout(Self) -> @device_materialize.DeviceWeightLayout' \
  'pub fn StaticOperation::operation_execution_digest(Self) -> @numeric_contract.OperationExecutionDigest'; do
  if ! rg -F -q "$required_api" \
    "$materialize_dir/pkg.generated.mbti" "$device_plan_dir/pkg.generated.mbti"; then
    fail "generated numeric-layout API drifted: $required_api"
  fi
done

forbidden_constructor='^pub fn (DeviceWeightRegion|DeviceWeightLayout|DeviceWeights|StaticOperation|StaticDevicePlan)::(new|from|wrap|revalidate)'
if ! printf '%s\n' 'pub fn DeviceWeightLayout::from_raw() -> DeviceWeightLayout' | \
  rg -q "$forbidden_constructor"; then
  fail "constructor positive control is ineffective"
fi
if rg -n "$forbidden_constructor" \
  "$materialize_dir/pkg.generated.mbti" "$device_plan_dir/pkg.generated.mbti"; then
  fail "opaque validated layout or static plan acquired a public constructor"
fi

profile_raw_constructor='^pub fn (DeviceExecutionProfile|OperationExecutionProfile|ActivationValueView|ExecutionRegionView|FinalOutputView|FinalRowView|ExactExecutionShape|ExactKernelShape|ProfileRegionId)::(new|from_raw|wrap|revalidate)'
if ! printf '%s\n' 'pub fn FinalOutputView::from_raw() -> FinalOutputView' | \
  rg -q "$profile_raw_constructor"; then
  fail "profile raw-constructor positive control is ineffective"
fi
if rg -n "$profile_raw_constructor" "$profile_dir/pkg.generated.mbti"; then
  fail "opaque validated profile view acquired a public raw constructor"
fi

profile_factory_api_is_allowed() {
  case "$1" in
    'pub fn ExactExecutionShape::full_prefill(batch_rows~ : Int, sequence_tokens~ : Int) -> Self raise DeviceProfileError'|\
      'pub fn ExactExecutionShape::full_recompute(batch_rows~ : Int, sequence_tokens~ : Int) -> Self raise DeviceProfileError'|\
      'pub fn ProfileRegionId::from_stable_int(Int) -> Self raise DeviceProfileError') return 0 ;;
    *) return 1 ;;
  esac
}

if profile_factory_api_is_allowed \
  'pub fn ExactExecutionShape::from_raw() -> Self'; then
  fail "profile factory allowlist positive control accepted a raw constructor"
fi
if ! profile_factory_api_is_allowed \
  'pub fn ProfileRegionId::from_stable_int(Int) -> Self raise DeviceProfileError'; then
  fail "profile factory allowlist positive control rejected a checked constructor"
fi
profile_factories=$(rg \
  '^pub fn (DeviceExecutionProfile|OperationExecutionProfile|ActivationValueView|ExecutionRegionView|FinalOutputView|FinalRowView|ExactExecutionShape|ExactKernelShape|ProfileRegionId)::' \
  "$profile_dir/pkg.generated.mbti" | rg -v '\(Self' || true)
while IFS= read -r factory; do
  [ -n "$factory" ] || continue
  profile_factory_api_is_allowed "$factory" || \
    fail "opaque profile type acquired an unapproved factory: $factory"
done <<EOF
$profile_factories
EOF

for file in $materialize_production $device_plan_production $profile_production; do
  lines=$(wc -l < "$file" | tr -d ' ')
  [ "$lines" -lt 500 ] || fail "$file exceeds the 499-line production budget"
done

echo "numeric device-weight layout and static-plan boundary: ok"
