#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

launch_dir="kernels/launch_contract"
interface="$launch_dir/pkg.generated.mbti"

fail() {
  echo "paged numeric-v4 launch boundary: $1" >&2
  exit 1
}

[ -d "$launch_dir" ] || fail "missing launch-contract package"
[ -f "$interface" ] || fail "missing generated launch-contract interface"

production=$(rg --files "$launch_dir" --glob '*.mbt' \
  --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt')
[ -n "$production" ] || fail "production discovery is empty"

production_has() {
  pattern=$1
  shift
  rg -F -q "$pattern" "$@"
}

for required in \
  'pub fn admit_paged_graph_v4(' \
  'model_plan.kv_execution() != PagedKeyValue' \
  'resolved.catalog_version() != @catalog.CatalogVersion::v4()' \
  'binding.semantic_version() != @catalog.KernelSemanticVersion::v4()' \
  'binding.operation_execution_digest() != Some(digest)' \
  'binding.aot_entry_point() is Some(entry_point)' \
  'input.entry_point != entry_point' \
  'numeric_schema_digest: model_plan.numeric_binding().digest()' \
  '@numeric_contract.storage_byte_length(' \
  'WeightScaleInput(reference, scale)' \
  'parameter.scale() != Some(scale)' \
  'metadata.role() != ScaleMetadataTensor' \
  'profiles.length() > limits.max_contracts / operations.length()' \
  'let expected_contracts = profiles.length() * operations.length()' \
  'if inputs.length() != expected_contracts' \
  'let profile_index = input_index / operations.length()' \
  'let op_index = input_index % operations.length()' \
  'model_plan.numeric_binding().operations().get(operation.as_int())' \
  'profile.id.as_int() != index + 1'; do
  production_has "$required" $production || \
    fail "missing v4 admission invariant: $required"
done

# Validated ModelPlan creation normally rejects affine/codebook storage before
# this package can observe it. Keep the defensive launch rejection explicit and
# before the generic execution-storage check so its error remains meaningful.
metadata_line=$(rg -n -F \
  'if tensor.zero_point() is Some(_) || tensor.codebook() is Some(_)' \
  "$launch_dir/paged_v4_numeric.mbt" | sed -n '1s/:.*//p')
storage_line=$(rg -n -F \
  'if !execution.accepts_storage(tensor.storage())' \
  "$launch_dir/paged_v4_numeric.mbt" | sed -n '1s/:.*//p')
[ -n "$metadata_line" ] || fail "missing unsupported-metadata rejection"
[ -n "$storage_line" ] || fail "missing execution-storage rejection"
[ "$metadata_line" -lt "$storage_line" ] || \
  fail "unsupported numeric metadata is not rejected before storage matching"

# The maximum public input size is fixed at 32. Reject before defensive copying
# so hostile callers cannot force an unbounded allocation at a trust boundary.
bound_line=$(rg -n -F \
  'if operands.length() > ABSOLUTE_MAX_OPERANDS_PER_CONTRACT' \
  "$launch_dir/operands.mbt" | sed -n '1s/:.*//p')
copy_line=$(rg -n -F 'ReadOnlyArray::from_array(operands)' \
  "$launch_dir/operands.mbt" | sed -n '1s/:.*//p')
[ -n "$bound_line" ] && [ -n "$copy_line" ] || \
  fail "bounded owned-operand helper is incomplete"
[ "$bound_line" -lt "$copy_line" ] || \
  fail "input operands are copied before the absolute bound is checked"

# V4 uses canonical profile-major/operation-major indexing and direct operation
# IDs. These forbidden scans are positive-controlled to keep the complexity
# check effective while preserving the legacy v1/v3 implementation unchanged.
complexity_pattern='for earlier|operation_index\(|paged_profile_for\(|paged_v4_input_for\('
if ! printf '%s\n' 'for earlier in 0..<index {' | rg -q "$complexity_pattern"; then
  fail "complexity positive control is ineffective"
fi
if rg -n "$complexity_pattern" \
  "$launch_dir/paged_v4_admit.mbt" "$launch_dir/paged_v4_numeric.mbt"; then
  fail "v4 admission reacquired a quadratic or repeated linear lookup"
fi

allowed_import() {
  case "$1" in
    vectie/lunaflux/kernels/catalog|\
      vectie/lunaflux/kv/device_layout|\
      vectie/lunaflux/model/numeric_contract|\
      vectie/lunaflux/model/plan|\
      vectie/lunaflux/model/spec) return 0 ;;
    *) return 1 ;;
  esac
}

if allowed_import vectie/lunaflux/kernels/artifact; then
  fail "dependency allowlist positive control accepted artifact authority"
fi
if ! allowed_import vectie/lunaflux/model/numeric_contract; then
  fail "dependency allowlist positive control rejected numeric authority"
fi
imports=$(sed -n '1,/^}/p' "$launch_dir/moon.pkg" | \
  sed -n 's/^[[:space:]]*"\([^"]*\)".*/\1/p')
for imported in $imports; do
  allowed_import "$imported" || \
    fail "launch contract has an unauthorized production import: $imported"
done

forbidden_authority='@(artifact|artifact_file|device_materialize|device_plan|device_executor|device_step|bootstrap|tensor_parallel)|internal/(cuda|nccl)|extern "c"|open_context\(|load_module\(|launch_kernel\('
if ! printf '%s\n' '@artifact.KernelArtifactBundle' | \
  rg -q "$forbidden_authority"; then
  fail "authority positive control is ineffective"
fi
if rg -n "$forbidden_authority" \
  "$launch_dir/paged_v4_admit.mbt" \
  "$launch_dir/paged_v4_numeric.mbt" \
  "$launch_dir/paged_v4_types.mbt"; then
  fail "v4 launch admission acquired loader, execution, or bootstrap authority"
fi

# Only the exact schema-v4 I8 bridge and schema-v5 FP8 reconstructor may carry
# this ABI across the launch-contract package boundary. The FP8 files rebuild
# inert, digest-joined recipes; they do not receive device or executor handles.
approved_manifest_v4_consumer() {
  case "$1" in
    kernels/artifact_file/load.mbt|\
      engine/execution_manifest_file/i8_v4_derive.mbt|\
      engine/execution_manifest_file/types.mbt|\
      engine/execution_manifest_file/fp8_v5_derive.mbt|\
      engine/execution_manifest_file/fp8_v5_load.mbt|\
      engine/execution_manifest_file/fp8_v5_types.mbt) return 0 ;;
    *) return 1 ;;
  esac
}

if approved_manifest_v4_consumer \
  engine/execution_manifest_file/fp8_v5_derive_escape.mbt; then
  fail "exact-consumer allowlist positive control accepted a near-name escape"
fi
if ! approved_manifest_v4_consumer \
  engine/execution_manifest_file/fp8_v5_derive.mbt; then
  fail "exact-consumer allowlist positive control rejected FP8 v5 reconstruction"
fi

legacy_consumers=$(rg -l 'PagedV4|admit_paged_graph_v4|WeightScaleInput' \
  kernels/artifact_file \
  kernels/tensor_parallel_launch_contract \
  engine/execution_manifest_file engine/device_worker_bootstrap \
  engine/tensor_parallel_execution_plan engine/tensor_parallel_device_worker \
  --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt' || true)
for consumer in $legacy_consumers; do
  approved_manifest_v4_consumer "$consumer" ||
    fail "artifact-file, manifest, bootstrap, or tensor-parallel production claims v4: $consumer"
done

inert_escape='@device_executor|@device_worker|extern "c"|open_context\(|load_module\(|launch_kernel\('
if ! printf '%s\n' 'load_module(' | rg -q "$inert_escape"; then
  fail "FP8 inert-reconstructor authority positive control is ineffective"
fi
if rg -n "$inert_escape" \
  engine/execution_manifest_file/fp8_v5_derive.mbt \
  engine/execution_manifest_file/fp8_v5_load.mbt \
  engine/execution_manifest_file/fp8_v5_types.mbt; then
  fail "FP8 v5 manifest reconstruction acquired device or execution authority"
fi

for check in \
  'kernels/launch_contract/admit.mbt:WeightScaleInput(_, _) => false' \
  'kernels/launch_contract/admit.mbt:resolved.catalog_version() != @catalog.CatalogVersion::v1()' \
  'kernels/launch_contract/paged_admit.mbt:if operand.role is WeightScaleInput(_, _)' \
  'kernels/launch_contract/paged_admit.mbt:WeightScaleInput(_, _) =>' \
  'engine/device_executor/blueprint.mbt:WeightScaleInput(_, _) => None' \
  'engine/device_executor/blueprint.mbt:contracts.catalog_version() != @catalog.CatalogVersion::v1()' \
  'engine/device_step/blueprint_admit.mbt:WeightScaleInput(_, _) =>' \
  'engine/device_step/bootstrap_encode.mbt:WeightScaleInput(_, _) => raise InvalidBootstrap(Blueprint)' \
  'kernels/luna_artifact_admission/phase5_execution_fixture_test.mbt:@catalog.CatalogEntry::new_paged_v3(' \
  'kernels/luna_artifact_admission/phase5_execution_fixture_test.mbt:@catalog.KernelCatalog::new(version=@catalog.CatalogVersion::v3(), entries~).resolve_paged(' \
  'kernels/artifact/admit.mbt:ensure_stateless_catalog_version(contracts.catalog_version())' \
  'kernels/artifact/admit.mbt:let expected = @catalog.CatalogVersion::v1()'; do
  file=${check%%:*}
  value=${check#*:}
  production_has "$value" "$file" || \
    fail "legacy fail-closed invariant drifted: $check"
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

if opaque_interface_is_safe 'pub struct HostileLaunch(Int)'; then
  fail "opacity positive control accepted a tuple representation"
fi
if opaque_interface_is_safe 'pub struct HostileLaunch {'; then
  fail "opacity positive control accepted exposed fields"
fi

for type in \
  KernelProfileId AotLaunchDimensions AotOperand LaunchContractLimits \
  KernelProfile PagedKernelProfile AotLaunchContractInput \
  PagedKvAotLaunchContractInput AotLaunchContract LaunchContractSet \
  PagedKvAotLaunchContract PagedKvLaunchContractSet \
  PagedV4AotLaunchContractInput PagedV4AotLaunchContract \
  PagedV4LaunchContractSet; do
  declaration=$(sed -n "/^pub struct ${type} {$/,/^} derive/p" "$interface")
  opaque_interface_is_safe "$declaration" || \
    fail "generated validated interface is not opaque: $type"
done

for input_api in \
  'pub fn AotLaunchContractInput::new' \
  'pub fn PagedKvAotLaunchContractInput::new' \
  'pub fn PagedV4AotLaunchContractInput::new'; do
  rg -F "$input_api" "$interface" | rg -F -q 'raise LaunchContractError' || \
    fail "checked input factory drifted: $input_api"
done

opaque_factory_is_allowed() {
  case "$1" in
    'pub fn KernelProfileId::from_stable_int(Int) -> Self raise LaunchContractError'|\
      'pub fn AotLaunchDimensions::new(grid_x~ : Int, grid_y~ : Int, grid_z~ : Int, block_x~ : Int, block_y~ : Int, block_z~ : Int, shared_memory_bytes~ : Int) -> Self raise LaunchContractError'|\
      'pub fn AotOperand::new(role~ : AotOperandRole, byte_count~ : Int64, alignment~ : Int64) -> Self raise LaunchContractError'|\
      'pub fn LaunchContractLimits::new(max_profiles~ : Int, max_contracts~ : Int, max_operands_per_contract~ : Int) -> Self raise LaunchContractError'|\
      'pub fn KernelProfile::full_prefill(id~ : KernelProfileId, batch_rows~ : Int, sequence_tokens~ : Int, token_rows~ : Int64) -> Self raise LaunchContractError'|\
      'pub fn PagedKernelProfile::new(id~ : KernelProfileId, max_query_rows~ : Int, max_query_tokens~ : Int, max_page_table_entries~ : Int) -> Self raise LaunchContractError'|\
      'pub fn AotLaunchContractInput::new(profile_id~ : KernelProfileId, operation_id~ : @plan.OperationId, entry_point~ : @catalog.AotKernelEntryPoint, dimensions~ : AotLaunchDimensions, operands~ : ArrayView[AotOperand]) -> Self raise LaunchContractError'|\
      'pub fn PagedKvAotLaunchContractInput::new(profile_id~ : KernelProfileId, operation_id~ : @plan.OperationId, entry_point~ : @catalog.AotKernelEntryPoint, dimensions~ : AotLaunchDimensions, operands~ : ArrayView[AotOperand]) -> Self raise LaunchContractError'|\
      'pub fn PagedV4AotLaunchContractInput::new(profile_id~ : KernelProfileId, operation_id~ : @plan.OperationId, entry_point~ : @catalog.AotKernelEntryPoint, dimensions~ : AotLaunchDimensions, operands~ : ArrayView[AotOperand]) -> Self raise LaunchContractError') return 0 ;;
    *) return 1 ;;
  esac
}

if opaque_factory_is_allowed \
  'pub fn AotOperand::from_raw() -> AotOperand'; then
  fail "opaque factory allowlist positive control accepted a raw factory"
fi
if ! opaque_factory_is_allowed \
  'pub fn KernelProfileId::from_stable_int(Int) -> Self raise LaunchContractError'; then
  fail "opaque factory allowlist positive control rejected a checked factory"
fi
opaque_factories=$(rg \
  '^pub fn (KernelProfileId|AotLaunchDimensions|AotOperand|LaunchContractLimits|KernelProfile|PagedKernelProfile|AotLaunchContractInput|PagedKvAotLaunchContractInput|AotLaunchContract|LaunchContractSet|PagedKvAotLaunchContract|PagedKvLaunchContractSet|PagedV4AotLaunchContractInput|PagedV4AotLaunchContract|PagedV4LaunchContractSet)::' \
  "$interface" | rg -v '\(Self' || true)
while IFS= read -r factory; do
  [ -n "$factory" ] || continue
  opaque_factory_is_allowed "$factory" || \
    fail "opaque launch type acquired an unapproved factory: $factory"
done <<EOF
$opaque_factories
EOF

forbidden_constructor='^pub fn (AotLaunchContract|LaunchContractSet|PagedKvAotLaunchContract|PagedKvLaunchContractSet|PagedV4AotLaunchContract|PagedV4LaunchContractSet)::(new|from|wrap|revalidate)'
if ! printf '%s\n' \
  'pub fn PagedV4LaunchContractSet::from_raw() -> PagedV4LaunchContractSet' | \
  rg -q "$forbidden_constructor"; then
  fail "result-constructor positive control is ineffective"
fi
if rg -n "$forbidden_constructor" "$interface"; then
  fail "validated launch result acquired a public constructor"
fi

rg -F -q \
  'WeightScaleInput(@plan.TensorRef, @plan.ScaleTensorRef)' "$interface" || \
  fail "generated semantic role lost exact scale association"
if rg -n 'PagedContractScope|KvSubsequence|scope\(' \
  "$launch_dir/paged_v4_types.mbt" \
  "$launch_dir/paged_v4_admit.mbt" \
  "$launch_dir/paged_v4_numeric.mbt"; then
  fail "v4 API acquired a partial-KV scope"
fi

for file in $production; do
  lines=$(wc -l < "$file" | tr -d ' ')
  [ "$lines" -lt 500 ] || fail "$file exceeds the 499-line production budget"
done

for statement in \
  'full paged graph' \
  'immediately after' \
  'numeric schema digest' \
  'does not admit an artifact' \
  'does not claim device execution' \
  'does not claim readiness'; do
  rg -i -F -q "$statement" "$launch_dir/README.mbt.md" || \
    fail "launch documentation is missing inert v4 statement: $statement"
done

echo "paged numeric-v4 launch boundary: ok"
