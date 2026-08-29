#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

fail() {
  echo "paged-v4 artifact boundary: $1" >&2
  exit 1
}

artifact_dir="kernels/artifact"
artifact_interface="$artifact_dir/pkg.generated.mbti"
memory_interface="engine/device_memory/pkg.generated.mbti"
device_interface="device/pkg.generated.mbti"
i8_interface="kernels/i8_inert_capability_admission/pkg.generated.mbti"
file_interface="kernels/artifact_file/pkg.generated.mbti"

for required_file in \
  "$artifact_interface" "$memory_interface" "$device_interface" \
  "$i8_interface" "$file_interface"; do
  [ -f "$required_file" ] || fail "missing generated interface: $required_file"
done

production=$(rg --files kernels/artifact kernels/artifact_file \
  engine/device_memory device kernels/i8_inert_capability_admission \
  --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt')
[ -n "$production" ] || fail "production discovery is empty"

production_has() {
  pattern=$1
  shift
  rg -F -q "$pattern" "$@"
}

adapter="$artifact_dir/paged_v4_admit.mbt"
for invariant in \
  'pub fn admit_paged_v4(' \
  'contracts : @launch_contract.PagedV4LaunchContractSet' \
  'ensure_paged_v4_artifact_contract_count(contracts.contracts().length())' \
  'const ABSOLUTE_MAX_PAGED_V4_ARTIFACT_CONTRACTS : Int = 1024' \
  'ensure_paged_v4_catalog_version(contracts.catalog_version())' \
  'let expected = @catalog.CatalogVersion::v4()' \
  'for contract in contracts.contracts()' \
  'append_required_entry_point(' \
  'contracts.model_identity()' \
  'contracts.device_target()' \
  'contracts.catalog_version()' \
  'admit_required('; do
  production_has "$invariant" "$adapter" || \
    fail "missing exact catalog-v4 adapter invariant: $invariant"
done

bound_line=$(rg -n -F \
  'ensure_paged_v4_artifact_contract_count(contracts.contracts().length())' \
  "$adapter" | sed -n '1s/:.*//p')
scan_line=$(rg -n -F 'for contract in contracts.contracts()' \
  "$adapter" | sed -n '1s/:.*//p')
allocation_line=$(rg -n -F \
  'let required_entries : Array[@catalog.AotKernelEntryPoint] = []' \
  "$adapter" | sed -n '1s/:.*//p')
if ! printf '%s\n' '1 2 3' | rg -q '1 2 3'; then
  fail "work-bound ordering positive control is ineffective"
fi
[ -n "$bound_line" ] && [ -n "$scan_line" ] && [ -n "$allocation_line" ] || \
  fail "catalog-v4 total-work bound or first scan is missing"
[ "$bound_line" -lt "$scan_line" ] && [ "$bound_line" -lt "$allocation_line" ] || \
  fail "catalog-v4 total-work bound does not precede allocation and scanning"

[ "$(rg -F -c 'admit_required(' "$adapter")" -eq 1 ] || \
  fail "catalog-v4 adapter must delegate exactly once"

delegated_checks='exact_digest|validate_module_inputs|validate_entry_inputs|own_module_input|Bytes::from_array|sha256\(|validate_symbol|DuplicateModule|UnreferencedModule'
if ! printf '%s\n' 'validate_module_inputs' | rg -q "$delegated_checks"; then
  fail "delegation positive control is ineffective"
fi
if rg -n "$delegated_checks" "$adapter"; then
  fail "catalog-v4 adapter duplicated shared artifact validation"
fi

for legacy in \
  'kernels/artifact/admit.mbt:let expected = @catalog.CatalogVersion::v1()' \
  'kernels/artifact/admit.mbt:let expected = @catalog.CatalogVersion::v3()' \
  'kernels/artifact/admit.mbt:ensure_stateless_catalog_version(contracts.catalog_version())' \
  'kernels/artifact/admit.mbt:ensure_paged_catalog_version(contracts.catalog_version())'; do
  file=${legacy%%:*}
  value=${legacy#*:}
  production_has "$value" "$file" || fail "legacy fail-closed check drifted: $legacy"
done

if ! printf '%s\n' \
  'artifacts.catalog_version() == contracts.catalog_version()' | \
  rg -q 'catalog_version\(\) == contracts\.catalog_version\(\)'; then
  fail "cross-version guard positive control is ineffective"
fi
for cross_version_guard in \
  'engine/device_executor/blueprint.mbt:static_plan.catalog_version() != @catalog.CatalogVersion::v1()' \
  'engine/device_executor/blueprint.mbt:contracts.catalog_version() != @catalog.CatalogVersion::v1()' \
  'engine/device_executor/blueprint.mbt:artifacts.catalog_version() != @catalog.CatalogVersion::v1()' \
  'engine/device_step/blueprint_admit.mbt:let catalog = @catalog.CatalogVersion::v3()' \
  'engine/device_step/blueprint_admit.mbt:contracts.catalog_version() != catalog' \
  'engine/device_step/blueprint_admit.mbt:artifacts.catalog_version() != catalog' \
  'engine/device_step/bootstrap_admit.mbt:blueprint.catalog_version() != @catalog.CatalogVersion::v3()' \
  'engine/device_step/bootstrap_admit.mbt:artifacts.catalog_version() != blueprint.catalog_version()' \
  'kernels/artifact/reauthenticate.mbt:self.catalog_version != @catalog.CatalogVersion::v3()' \
  'engine/tensor_parallel_execution_plan/validate.mbt:artifacts.authenticates_tensor_parallel(' \
  'engine/execution_manifest_file/reader.mbt:@catalog.CatalogVersion::v3().as_int()' \
  'engine/execution_manifest_file/load.mbt:@artifact_file.load_paged_kv_admitted('; do
  guard_file=${cross_version_guard%%:*}
  guard_value=${cross_version_guard#*:}
  production_has "$guard_value" "$guard_file" || \
    fail "generic bundle cross-version guard drifted: $cross_version_guard"
done

approved_v4_consumer() {
  case "$1" in
    kernels/launch_contract/*|\
      kernels/artifact/paged_v4_admit.mbt|\
      kernels/artifact_file/load.mbt|\
      kernels/fp8_launch_abi/admit.mbt|\
      kernels/fp8_launch_abi/types.mbt|\
      kernels/numeric_capability_manifest/paged_v4.mbt|\
      engine/device_step/i8_blueprint_admit.mbt|\
      engine/device_step/fp8_blueprint_admit.mbt|\
      engine/execution_manifest_file/i8_v4_derive.mbt|\
      engine/execution_manifest_file/types.mbt|\
      engine/execution_manifest_file/fp8_v5_derive.mbt|\
      engine/execution_manifest_file/fp8_v5_load.mbt|\
      engine/execution_manifest_file/fp8_v5_types.mbt|\
      tests/i8_weight_scale_cuda_probe/contract_admission.mbt|\
      tests/i8_weight_scale_cuda_probe/contract_operands.mbt) return 0 ;;
    *) return 1 ;;
  esac
}

if approved_v4_consumer engine/device_step/fp8_blueprint_admit_escape.mbt; then
  fail "catalog-v4 allowlist positive control accepted a near-name consumer"
fi
if ! approved_v4_consumer engine/device_step/fp8_blueprint_admit.mbt; then
  fail "catalog-v4 allowlist positive control rejected the exact FP8 blueprint"
fi

v4_consumers=$(rg -l 'PagedV4LaunchContractSet|admit_paged_graph_v4' \
  --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt' || true)
for consumer in $v4_consumers; do
  approved_v4_consumer "$consumer" ||
    fail "catalog-v4 launch authority escaped its narrow consumer allowlist: $consumer"
done

v4_escape=$(rg -l 'PagedV4|admit_paged_v4' \
  kernels/artifact_file engine/device_executor engine/device_step \
  engine/device_worker_bootstrap engine/execution_manifest_file \
  kernels/tensor_parallel_launch_contract \
  --glob '*.mbt' --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt' || true)
for consumer in $v4_escape; do
  approved_v4_consumer "$consumer" ||
    fail "legacy consumer acquired catalog-v4 artifact authority: $consumer"
done

fp8_blueprint_escape='@artifact_file|@execution_manifest_file|extern "c"|open_context\(|load_module\(|launch_kernel\('
if ! printf '%s\n' '@artifact_file.load(' | rg -q "$fp8_blueprint_escape"; then
  fail "FP8 blueprint authority positive control is ineffective"
fi
if rg -n "$fp8_blueprint_escape" engine/device_step/fp8_blueprint_admit.mbt; then
  fail "FP8 blueprint acquired file-loader, context, module, or launch authority"
fi

forbidden_authority='@(artifact_file|device_materialize|device_plan|device_executor|device_step|bootstrap|execution_manifest|tensor_parallel)|internal/(cuda|nccl)|extern "c"|load_module\(|launch_kernel\('
if ! printf '%s\n' 'load_module(' | rg -q "$forbidden_authority"; then
  fail "authority positive control is ineffective"
fi
if rg -n "$forbidden_authority" "$adapter"; then
  fail "catalog-v4 artifact adapter acquired loader or execution authority"
fi

opaque_interface() {
  interface=$1
  type=$2
  declaration=$(awk -v header="pub struct ${type} {" '
    $0 == header { seen = 1 }
    seen { print }
    seen && /^}/ { exit }
  ' "$interface")
  [ -n "$declaration" ] || return 1
  printf '%s\n' "$declaration" | \
    rg -F -q "pub struct ${type} {" || return 1
  printf '%s\n' "$declaration" | \
    rg -F -q '  // private fields' || return 1
  if printf '%s\n' "$declaration" | rg -q '^  [A-Za-z_][A-Za-z0-9_]* :'; then
    return 1
  fi
  return 0
}

if opaque_interface /dev/stdin HostileTuple <<'EOF'
pub struct HostileTuple(Int)
EOF
then
  fail "opacity positive control accepted a tuple representation"
fi
if opaque_interface /dev/stdin HostileFields <<'EOF'
pub struct HostileFields {
  value : Int
}
EOF
then
  fail "opacity positive control accepted public fields"
fi

for pair in \
  "$artifact_interface:KernelArtifactBundle" \
  "$artifact_interface:KernelModule" \
  "$artifact_interface:KernelEntryPoint" \
  "$artifact_interface:KernelModuleInput" \
  "$artifact_interface:KernelEntryPointInput" \
  "$artifact_interface:KernelArtifactLimits" \
  "$file_interface:KernelManifestDigest" \
  "$file_interface:KernelArtifactFileLimits" \
  "$memory_interface:DeviceMemoryPlan" \
  "$memory_interface:ActivationRegion" \
  "$memory_interface:WorkspaceRegion" \
  "$memory_interface:OperationMemoryBinding" \
  "$memory_interface:DeviceMemoryLimits" \
  "$memory_interface:ActivationSlotId" \
  "$memory_interface:DeviceMemoryArena" \
  "$device_interface:DeviceNumericCapability" \
  "$device_interface:DeviceNumericCapabilityVersion" \
  "$device_interface:DeviceNumericCapabilityDigest" \
  "$i8_interface:I8InertKernelCapabilityAdmission" \
  "$i8_interface:I8ProductionPolicyAdmission" \
  "$i8_interface:I8InertKernelCapabilityAdmissionVersion" \
  "$i8_interface:I8InertKernelCapabilityAdmissionDigest"; do
  interface=${pair%%:*}
  type=${pair#*:}
  opaque_interface "$interface" "$type" || \
    fail "generated validated interface is forgeable: $type"
done

free_authority_mint_is_allowed() {
  case "$1" in
    'pub fn admit(@launch_contract.LaunchContractSet, ArrayView[KernelModuleInput], ArrayView[KernelEntryPointInput], KernelArtifactLimits) -> KernelArtifactBundle raise KernelArtifactError'|\
      'pub fn admit_paged_kv(@launch_contract.PagedKvLaunchContractSet, ArrayView[KernelModuleInput], ArrayView[KernelEntryPointInput], KernelArtifactLimits) -> KernelArtifactBundle raise KernelArtifactError'|\
      'pub fn admit_paged_v4(@launch_contract.PagedV4LaunchContractSet, ArrayView[KernelModuleInput], ArrayView[KernelEntryPointInput], KernelArtifactLimits) -> KernelArtifactBundle raise KernelArtifactError'|\
      'pub fn admit_tensor_parallel(@tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet, ArrayView[KernelModuleInput], ArrayView[KernelEntryPointInput], KernelArtifactLimits) -> KernelArtifactBundle raise KernelArtifactError'|\
      'pub fn allocate(@device.Context, DeviceMemoryPlan) -> DeviceMemoryArena raise DeviceMemoryError'|\
      'pub fn plan(@device_plan.StaticDevicePlan, DeviceMemoryLimits) -> DeviceMemoryPlan raise DeviceMemoryError'|\
      'pub fn admit_finite_fp8_e4m3_w8a8(DeviceCapability) -> DeviceNumericCapability raise DeviceNumericCapabilityError'|\
      'pub fn admit_symmetric_i8_weight_only_per_output_channel_v1(DeviceCapability) -> DeviceNumericCapability raise DeviceNumericCapabilityError'|\
      'pub fn admit_catalog_only_symmetric_i8_weight_only_v1(@plan.ModelPlan, @device.DeviceNumericCapability, @catalog.ResolvedKernelCatalog, I8InertKernelCapabilityAdmissionLimits) -> I8InertKernelCapabilityAdmission raise I8InertKernelCapabilityAdmissionError'|\
      'pub fn admit_production_llama_symmetric_i8_weight_only_v1(@plan.ModelPlan, @device.DeviceNumericCapability, @catalog.ResolvedKernelCatalog, I8InertKernelCapabilityAdmissionLimits) -> I8ProductionPolicyAdmission raise I8InertKernelCapabilityAdmissionError'|\
      'pub fn preflight_production_llama_symmetric_i8_weight_only_v1(@plan.ModelPlan) -> Unit raise I8InertKernelCapabilityAdmissionError') return 0 ;;
    *) return 1 ;;
  esac
}

if free_authority_mint_is_allowed \
  'pub fn forge_bundle() -> KernelArtifactBundle'; then
  fail "free authority-mint positive control accepted a raw forge"
fi
if ! free_authority_mint_is_allowed \
  'pub fn plan(@device_plan.StaticDevicePlan, DeviceMemoryLimits) -> DeviceMemoryPlan raise DeviceMemoryError'; then
  fail "free authority-mint positive control rejected an approved admission"
fi

audited_return_pattern=' -> .*(KernelArtifactBundle|KernelModule|KernelEntryPoint|KernelModuleInput|KernelEntryPointInput|KernelArtifactLimits|DeviceMemoryPlan|ActivationRegion|WorkspaceRegion|OperationMemoryBinding|DeviceMemoryLimits|ActivationSlotId|DeviceMemoryArena|DeviceNumericCapability|DeviceNumericCapabilityVersion|DeviceNumericCapabilityDigest|I8InertKernelCapabilityAdmission|I8InertKernelCapabilityAdmissionVersion|I8InertKernelCapabilityAdmissionDigest|I8ProductionPolicyAdmission)'
if ! printf '%s\n' \
  'pub fn forge_optional() -> KernelArtifactBundle?' | \
  rg -q "$audited_return_pattern"; then
  fail "free authority-mint discovery missed an optional hostile mint"
fi
free_authority_mints=$(rg --no-filename \
  '^pub fn [a-z][A-Za-z0-9_]*\(' \
  "$artifact_interface" "$memory_interface" "$device_interface" \
  "$i8_interface" | rg "$audited_return_pattern" || true)
while IFS= read -r mint; do
  [ -n "$mint" ] || continue
  free_authority_mint_is_allowed "$mint" || \
    fail "audited opaque authority acquired an unapproved free mint: $mint"
done <<EOF
$free_authority_mints
EOF

opaque_factory_is_allowed() {
  case "$1" in
    'pub fn KernelModuleInput::new(digest~ : @catalog.AotArtifactDigest, module_bytes~ : FixedArray[Byte]) -> Self'|\
      'pub fn KernelModuleInput::from_bytes(digest~ : @catalog.AotArtifactDigest, module_bytes~ : Bytes) -> Self'|\
      'pub fn KernelEntryPointInput::new(entry_point~ : @catalog.AotKernelEntryPoint, function_symbol~ : String) -> Self'|\
      'pub fn KernelArtifactLimits::new(max_modules~ : Int, max_entry_points~ : Int, max_module_bytes~ : Int64, max_total_module_bytes~ : Int64, max_function_symbol_bytes~ : Int) -> Self raise KernelArtifactError'|\
      'pub fn KernelArtifactFileLimits::new(max_manifest_bytes~ : Int64, max_json_depth~ : Int, max_path_bytes~ : Int, artifact~ : @artifact.KernelArtifactLimits) -> Self raise KernelArtifactFileError'|\
      'pub fn KernelManifestDigest::from_sha256(String) -> Self raise KernelArtifactFileError'|\
      'pub fn DeviceMemoryLimits::new(activation_alignment~ : Int64, max_arena_bytes~ : Int64) -> Self raise DeviceMemoryError'|\
      'pub fn DeviceNumericCapabilityVersion::v1() -> Self'|\
      'pub fn I8InertKernelCapabilityAdmissionVersion::v1() -> Self') return 0 ;;
    *) return 1 ;;
  esac
}

if opaque_factory_is_allowed \
  'pub fn KernelArtifactBundle::from_raw() -> Self'; then
  fail "opaque factory allowlist positive control accepted from_raw"
fi
if ! opaque_factory_is_allowed \
  'pub fn DeviceNumericCapabilityVersion::v1() -> Self'; then
  fail "opaque factory allowlist positive control rejected an approved factory"
fi

opaque_factories=$(rg --no-filename \
  '^pub fn (KernelArtifactBundle|KernelModule|KernelEntryPoint|KernelModuleInput|KernelEntryPointInput|KernelArtifactLimits|KernelArtifactFileLimits|KernelManifestDigest|DeviceMemoryPlan|ActivationRegion|WorkspaceRegion|OperationMemoryBinding|DeviceMemoryLimits|ActivationSlotId|DeviceMemoryArena|DeviceNumericCapabilityVersion|DeviceNumericCapabilityDigest|I8InertKernelCapabilityAdmissionVersion|I8InertKernelCapabilityAdmissionDigest)::' \
  "$artifact_interface" "$file_interface" "$memory_interface" \
  "$device_interface" "$i8_interface" | rg -v '\(Self' || true)
while IFS= read -r factory; do
  [ -n "$factory" ] || continue
  opaque_factory_is_allowed "$factory" || \
    fail "audited opaque type acquired an unapproved factory: $factory"
done <<EOF
$opaque_factories
EOF

if rg -n '^pub enum KernelModuleInput|^pub struct KernelManifestDigest\(' \
  "$artifact_interface" "$file_interface"; then
  fail "artifact input or manifest digest regained a forgeable representation"
fi

rg -F -q \
  'pub fn admit_paged_v4(@launch_contract.PagedV4LaunchContractSet, ArrayView[KernelModuleInput], ArrayView[KernelEntryPointInput], KernelArtifactLimits) -> KernelArtifactBundle raise KernelArtifactError' \
  "$artifact_interface" || fail "generated catalog-v4 artifact API drifted"

for source_file in $production; do
  lines=$(wc -l < "$source_file" | tr -d ' ')
  [ "$lines" -lt 500 ] || fail "$source_file exceeds the 499-line budget"
done

for statement in \
  'exact first-occurrence' 'content-digest' 'does not read' \
  'launch a kernel' 'physical device readiness'; do
  rg -i -F -q "$statement" "$artifact_dir/README.mbt.md" || \
    fail "artifact documentation is missing inert v4 statement: $statement"
done
for debt_statement in \
  'artifact-v4 admission subphase owns' \
  'remove the cap before D3 executor' \
  'latest removal phase'; do
  rg -F -q "$debt_statement" "$artifact_dir/README.mbt.md" || \
    fail "bounded-comparison debt lacks owner/removal/deadline: $debt_statement"
done

echo "paged-v4 artifact boundary: ok"
