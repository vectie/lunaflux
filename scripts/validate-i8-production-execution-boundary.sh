#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

fail() {
  echo "I8 production execution boundary: $1" >&2
  exit 1
}

i8_interface="kernels/i8_inert_capability_admission/pkg.generated.mbti"
step_interface="engine/device_step/pkg.generated.mbti"
manifest_interface="engine/execution_manifest_file/pkg.generated.mbti"
loader_interface="model/numeric_weight_file/pkg.generated.mbti"
for interface in \
  "$i8_interface" "$step_interface" "$manifest_interface" \
  "$loader_interface"; do
  [ -f "$interface" ] || fail "missing generated interface: $interface"
done

policy="kernels/i8_inert_capability_admission/production_policy.mbt"
load="engine/execution_manifest_file/i8_v4_load.mbt"
schema="engine/execution_manifest_file/i8_v4_schema.mbt"
join="engine/device_step/i8_loaded_join.mbt"
bootstrap="engine/device_step/i8_bootstrap_v3.mbt"
executor="engine/device_step/i8_executor_prepare.mbt"

for invariant in \
  'validate_production_llama_graph(plan)' \
  'let eligible = i8_operation_kind_allowed(operation.kind())' \
  'if !has_parameter ||' \
  '!exact_i8_weight_only_execution(execution)' \
  'MixedFp8Execution' \
  'tensor.zero_point() is None' \
  'tensor.codebook() is None' \
  'scale_uses[scale_index] != 0' \
  'scale_tensor_count != i8_parameter_count'; do
  rg -F -q "$invariant" "$policy" || \
    fail "production policy invariant drifted: $invariant"
done

for invariant in \
  'require_nonnegative_int(root, "schema_version", Root) != 4' \
  '@catalog.CatalogVersion::v4().as_int()' \
  '"numeric_schema_sha256", "i8_admission_sha256", "device_numeric_sha256"' \
  '"weight_artifact_sha256"' \
  'IdentityMismatch(NumericSchemaDigest)' \
  'IdentityMismatch(DeviceNumericDigest)'; do
  rg -F -q "$invariant" "$schema" || \
    fail "schema-v4 exactness invariant drifted: $invariant"
done

preflight_line=$(rg -n -F \
  'preflight_production_llama_symmetric_i8_weight_only_v1(' "$load" | \
  sed -n '1s/:.*//p')
manifest_line=$(rg -n -F 'let bytes = read_manifest_snapshot(' "$load" | \
  sed -n '1s/:.*//p')
artifact_line=$(rg -n -F 'load_paged_v4_admitted(' "$load" | \
  sed -n '1s/:.*//p')
[ -n "$preflight_line" ] && [ -n "$manifest_line" ] && \
  [ -n "$artifact_line" ] || fail "admission ordering anchors are missing"
[ "$preflight_line" -lt "$manifest_line" ] && \
  [ "$manifest_line" -lt "$artifact_line" ] || \
  fail "model policy or schema admission no longer precedes artifact opening"
if ! printf '%s\n' 'preflight 1 read 2 artifact 3' | \
  rg -q 'preflight 1 read 2 artifact 3'; then
  fail "ordering positive control is ineffective"
fi

for invariant in \
  'ignore(weights.allocation() catch { _ => raise NumericWeightClose })' \
  'weights.identity() != blueprint.identity()' \
  'weights.numeric_schema_digest() != blueprint.numeric_schema_digest()' \
  'weights.layout() != blueprint.weight_layout()' \
  'weights.artifact_digest() != expected_weight_artifact_digest' \
  'let memory_plan = blueprint.memory_plan' \
  'let artifacts = blueprint.artifacts' \
  'priv memory_plan : @device_memory.DeviceMemoryPlan' \
  'priv artifacts : @artifact.KernelArtifactBundle' \
  'priv weights : @numeric_weight_file.AuthenticatedNumericWeightAuthority'; do
  rg -F -q "$invariant" "$join" engine/device_step/blueprint_types.mbt || \
    fail "loaded-weight four-way join drifted: $invariant"
done

join_signature=$(sed -n \
  '/^pub fn join_i8_loaded_execution_blueprint_v1(/,/^) -> I8LoadedExecutionBlueprint/p' \
  "$join")
[ -n "$join_signature" ] || fail "loaded-weight join signature is missing"
if printf '%s\n' "$join_signature" | \
  rg -q 'DeviceMemoryPlan|KernelArtifactBundle|memory_plan|artifacts'; then
  fail "loaded-weight join regained caller-substitutable memory/artifacts"
fi

bootstrap_signature=$(sed -n \
  '/^pub fn admit_i8_device_worker_bootstrap_v3(/,/^) -> I8DeviceWorkerBootstrapManifestV3/p' \
  "$bootstrap")
[ -n "$bootstrap_signature" ] || fail "bootstrap-v3 signature is missing"
if printf '%s\n' "$bootstrap_signature" | rg -q 'KernelArtifactBundle|artifacts'; then
  fail "bootstrap-v3 regained caller-substitutable artifacts"
fi
rg -F -q 'let artifacts = blueprint.artifacts' "$bootstrap" || \
  fail "bootstrap-v3 no longer uses the blueprint-retained artifact bundle"
rg -F -q 'contracts.profiles().length() != 1' \
  engine/device_step/i8_blueprint_admit.mbt || \
  fail "production-I8 v1 no longer requires one exact launch profile"

if rg -n 'memory_plan : @device_memory.DeviceMemoryPlan|artifacts : @artifact.KernelArtifactBundle' \
  "$executor"; then
  fail "executor preparation regained caller-substitutable memory/artifacts"
fi

for invariant in \
  'lunaflux.device-worker-bootstrap.v3\u{0000}' \
  'encoder.append_string(blueprint.numeric_schema_digest().as_hex())' \
  'encoder.append_string(blueprint.i8_admission_digest().as_hex())' \
  'encoder.append_string(blueprint.device_numeric_digest().as_hex())' \
  'WeightScaleInput(weight, scale)' \
  'encoder.append_string(step.operation_execution_digest().as_hex())'; do
  rg -F -q "$invariant" "$bootstrap" || \
    fail "bootstrap-v3 digest coverage drifted: $invariant"
done

test_source="engine/device_step/i8_executor_wbtest.mbt"
for invariant in \
  'run_i8_construction_stages(' \
  'run_paged_close_order(' \
  'i8_blueprint_fixture()' \
  'WeightScaleInput(weight, scale)' \
  'fixture.blueprint.memory_plan' \
  'fixture.blueprint.artifacts' \
  'run_paged_ordered_step_range(' \
  'ExecutorLaunchFailed('; do
  rg -F -q "$invariant" "$test_source" || \
    fail "I8 executor test lost production-seam anchor: $invariant"
done
for invariant in \
  'let stages : Array[() -> Unit raise DeviceStepError]' \
  'run_i8_construction_stages(' \
  'load_paged_modules(context, artifacts, resources)' \
  'load_paged_functions(functions, resources)' \
  'prepare_i8_paged_steps(' \
  'prepare_paged_ordered_executor(context, steps, resources, graph_policy)'; do
  rg -F -q "$invariant" "$executor" || \
    fail "production I8 construction is no longer tied to tested seam: $invariant"
done
for invariant in \
  'ordered.enqueue(index)' \
  'ordered.record_completion()' \
  'ordered.wait_completion()' \
  'ordered.reset()' \
  'abort_paged_ordered_executor(ordered)'; do
  rg -F -q "$invariant" engine/device_step/paged_executor_run.mbt || \
    fail "native paged ordered execution seam drifted: $invariant"
done
rg -F -q 'run_paged_close_order(' engine/device_step/paged_executor_cleanup.mbt || \
  fail "native paged cleanup no longer uses the tested close-order seam"
if rg -n 'fake_i8_construct_and_cleanup|FakeI8LaunchStep' "$test_source"; then
  fail "disconnected I8 integer/boolean toy test returned"
fi

if rg -n 'physical_execution_ready|cuda_ready' \
  engine/device_step/i8_*.mbt engine/execution_manifest_file/i8_v4_*.mbt \
  --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt'; then
  fail "I8 software path published unearned physical readiness authority"
fi
rg -F -q \
  'pub fn I8DeviceWorkerBootstrapManifestV3::worker_startup_contract(' \
  engine/device_step/bootstrap_types.mbt || \
  fail "bootstrap-v3 lost its sole process-startup binding"

for invariant in \
  'context.capability()' \
  'observed.digest() != blueprint.device_numeric_digest()' \
  'let weight = loaded.authenticated_allocation()' \
  'context.lease_allocation(weight)' \
  'WeightScaleInput(_, scale) => scale.tensor()' \
  'region.end_offset_bytes() <= weight.byte_count()' \
  'validate_i8_physical_regions(' \
  'load_paged_modules(context, artifacts, resources)' \
  'load_paged_functions(functions, resources)' \
  'resources.numeric_weights = Some(I8NumericWeights(loaded))' \
  'paged_construction_failed(resources, error)'; do
  rg -F -q "$invariant" "$executor" || \
    fail "I8 software executor invariant drifted: $invariant"
done

for invariant in \
  'I8NumericWeights(I8LoadedExecutionBlueprint)' \
  'Some(I8NumericWeights(owner))' \
  'owner.close()'; do
  rg -F -q "$invariant" \
    engine/device_step/paged_executor_types.mbt \
    engine/device_step/paged_executor_cleanup.mbt ||
    fail "I8 tagged numeric-owner cleanup invariant drifted: $invariant"
done
[ "$(rg -F -c 'resources.numeric_weights = Some(I8NumericWeights(loaded))' "$executor")" -eq 1 ] ||
  fail "I8 constructor must transfer the exact loaded owner exactly once"
wrong_i8_owner='Some\((Bf16NumericWeights|Fp8NumericWeights)\(loaded\)\)'
if ! printf '%s\n' 'Some(Bf16NumericWeights(loaded))' | rg -q "$wrong_i8_owner"; then
  fail "I8 owner-tag positive control is ineffective"
fi
if rg -n "$wrong_i8_owner" "$executor"; then
  fail "I8 constructor transferred its loaded authority under the wrong owner tag"
fi

if rg -n '^pub fn .*tensor_parallel.*i8|^pub fn .*i8.*tensor_parallel' \
  "$manifest_interface" engine/execution_manifest_file; then
  fail "unsupported schema-v4 tensor-parallel API became reachable"
fi
if ! printf '%s\n' 'pub fn load_tensor_parallel_i8_v4' | \
  rg -q 'tensor_parallel.*i8|i8.*tensor_parallel'; then
  fail "tensor-parallel absence positive control is ineffective"
fi

for legacy in \
  'engine/execution_manifest_file/reader.mbt:schema_version == 2' \
  'engine/execution_manifest_file/reader.mbt:schema_version == 3' \
  'engine/execution_manifest_file/reader.mbt:@catalog.CatalogVersion::v3().as_int()' \
  'engine/device_step/blueprint_admit.mbt:let catalog = @catalog.CatalogVersion::v3()' \
  'engine/device_step/bootstrap_encode.mbt:WeightScaleInput(_, _) => raise InvalidBootstrap(Blueprint)'; do
  legacy_file=${legacy%%:*}
  legacy_value=${legacy#*:}
  rg -F -q "$legacy_value" "$legacy_file" || \
    fail "legacy fail-closed invariant drifted: $legacy"
done

opaque_interface() {
  interface=$1
  type=$2
  declaration=$(sed -n "/^pub struct ${type} {$/,/^}/p" "$interface")
  [ -n "$declaration" ] || return 1
  printf '%s\n' "$declaration" | rg -F -q '  // private fields' || return 1
  ! printf '%s\n' "$declaration" | rg -q '^  [A-Za-z_][A-Za-z0-9_]* :'
}

if opaque_interface /dev/stdin Hostile <<'EOF'
pub struct Hostile {
  value : Int
}
EOF
then
  fail "opacity positive control accepted a public field"
fi

for pair in \
  "$i8_interface:I8ProductionPolicyAdmission" \
  "$step_interface:I8PagedExecutionStep" \
  "$step_interface:I8PagedExecutionBlueprint" \
  "$step_interface:I8DeviceWorkerBootstrapManifestV3" \
  "$step_interface:I8LoadedExecutionBlueprint" \
  "$manifest_interface:I8PagedExecutionAdmissionV4" \
  "$loader_interface:AuthenticatedNumericWeightAuthority"; do
  interface=${pair%%:*}
  type=${pair#*:}
  opaque_interface "$interface" "$type" || \
    fail "generated I8 authority is forgeable: $type"
done

if rg -n '^pub fn I8LoadedExecutionBlueprint::(allocation|weights|executor|ready)' \
  "$step_interface"; then
  fail "loaded prerequisite leaked allocation or readiness authority"
fi

for import_file in \
  kernels/i8_inert_capability_admission/moon.pkg \
  engine/execution_manifest_file/moon.pkg; do
  if rg -n 'internal/(cuda|nccl)|engine/device_executor' "$import_file"; then
    fail "I8 admission imported hardware execution authority: $import_file"
  fi
done

for hostile in \
  'production policy rejects partial reviewed-operation coverage' \
  'production policy rejects mixed FP8 execution before catalog work' \
  'schema-v4 parser rejects numeric substitution duplicate and op reorder' \
  'I8 native fake seam construction cleans every real boundary in reverse' \
  'I8 construction cleanup failure is retained and retryable' \
  'I8 admitted steps bind exact scale region immediately after each weight' \
  'I8 blueprint retains exact memory and artifact authorities' \
  'production I8 blueprint rejects any unselected launch profile' \
  'I8 bootstrap-v3 startup method binds exact digest and runtime envelope' \
  'I8 host launch canary uses admitted semantic order including scale steps' \
  'I8 host launch failure poisons before any later admitted operation' \
  'shared liveness guard accepts before close and rejects after close'; do
  rg -F -q "$hostile" \
    kernels/i8_inert_capability_admission/*test.mbt \
    engine/execution_manifest_file/*test.mbt \
    engine/device_step/*test.mbt \
    model/numeric_weight_file/*test.mbt || \
    fail "missing hostile or positive-control test: $hostile"
done

scripts/validate-i8-weight-scale-physical-probe.sh >/dev/null || \
  fail "scoped physical I8 weight+scale probe boundary failed"

echo "I8 production execution boundary: ok"
