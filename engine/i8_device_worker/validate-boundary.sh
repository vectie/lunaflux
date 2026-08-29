#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
pkg="$root/engine/i8_device_worker"
bootstrap="$root/engine/device_worker_bootstrap"
child="$root/engine/device_worker_child/run.mbt"
release="$root/runtime/i8_release_worker/release_prepare.mbt"
release_test="$root/runtime/i8_release_worker/release_prepare_wbtest.mbt"
release_mbti="$root/runtime/i8_release_worker/pkg.generated.mbti"
instance="$root/runtime/instance_admission"
instance_mbti="$instance/pkg.generated.mbti"
paged_cleanup="$root/engine/device_step/paged_executor_cleanup.mbt"
i8_executor_prepare="$root/engine/device_step/i8_executor_prepare.mbt"
i8_manifest_transfer="$root/engine/execution_manifest_file/i8_v4_executor.mbt"

public_api=$(rg --no-filename -o '^pub fn [A-Za-z0-9_:]+' "$pkg"/*.mbt |
  sed 's/^pub fn //' | sort)
expected_public_api='FailedI8DeviceWorkerPreparation::cause
FailedI8DeviceWorkerPreparation::cleanup_failure
FailedI8DeviceWorkerPreparation::is_closed
FailedI8DeviceWorkerPreparation::retry_close
I8DeviceWorkerOwner::close
I8DeviceWorkerOwner::execute_frame
I8DeviceWorkerOwner::lifecycle
I8DeviceWorkerOwner::readiness_contract
admit_plan
prepare'
if [ "$public_api" != "$expected_public_api" ]; then
  echo "i8 worker public factory/method allowlist changed" >&2
  exit 1
fi

resource_types='Context|Allocation|NumericDeviceWeights|AuthenticatedNumericWeightAuthority|I8LoadedExecutionBlueprint|PagedGraphExecutor|FailedPagedGraphPreparation'
if rg -n -U "^pub fn[^{]*->[^{]*($resource_types)" "$pkg"/*.mbt >/dev/null ||
  rg -n "^pub fn .*-> .*($resource_types)" "$pkg/pkg.generated.mbti" >/dev/null; then
  echo "i8 worker exposes a resource-bearing return type" >&2
  exit 1
fi

if ! rg -n '^pub fn admit_plan\(.*-> I8DeviceWorkerPlan ' "$pkg/pkg.generated.mbti" >/dev/null ||
  ! rg -n '^pub fn prepare\(.*-> I8DeviceWorkerPreparation ' "$pkg/pkg.generated.mbti" >/dev/null ||
  [ "$(rg -c '^pub fn .*-> I8DeviceWorkerPlan ' "$pkg/pkg.generated.mbti")" -ne 1 ] ||
  [ "$(rg -c '^pub fn .*-> I8DeviceWorkerPreparation ' "$pkg/pkg.generated.mbti")" -ne 1 ]; then
  echo "i8 worker MBTI factory return allowlist changed or is stale" >&2
  exit 1
fi

if rg -n 'pub fn .*\b(context|allocation|weights|executor|loaded)\b' "$pkg"/*.mbt >/dev/null; then
  echo "i8 worker leaks a resource-bearing public accessor" >&2
  exit 1
fi

if rg -n '@device\.(open_context|Context::)|load_inspected_file|join_loaded_weights|prepare_paged_i8_v4_executor' "$pkg/admit.mbt" >/dev/null; then
  echo "i8 inert admission opens or prepares a live resource" >&2
  exit 1
fi

for required in \
  'load_inspected_file' \
  'authenticate_exact_numeric_weights' \
  'join_loaded_weights' \
  'prepare_paged_i8_v4_executor' \
  'admit_symmetric_i8_weight_only_per_output_channel_v1'; do
  if ! rg -n "$required" "$pkg/prepare.mbt" >/dev/null; then
    echo "i8 live preparation is missing $required" >&2
    exit 1
  fi
done

for required in \
  'bind_loaded_executor_transfer' \
  'physical_equal\(loaded\.blueprint\(\), self\.blueprint\(\)\)' \
  'loaded\.bootstrap_manifest\(\).*self\.bootstrap_manifest\(\)' \
  'loaded\.weight_artifact_digest\(\).*self\.weight_artifact_digest\(\)' \
  'I8PagedExecutorTransferV4::close'; do
  if ! rg -n -U "$required" "$i8_manifest_transfer" >/dev/null; then
    echo "I8 executor transfer lacks exact retained evidence $required" >&2
    exit 1
  fi
done

lower_prepare=$(sed -n '/^pub fn prepare_i8_paged_graph_executor_v1(/,/^}/p' "$i8_executor_prepare")
if ! printf '%s\n' "$lower_prepare" |
  rg -n -U '(?s)let resources = empty_paged_resources\(\).*resources\.numeric_weights = Some\(I8NumericWeights\(loaded\)\).*admit_i8_owned_preparation_evidence' >/dev/null ||
  ! printf '%s\n' "$lower_prepare" | rg -q 'paged_construction_failed\(resources, error\)'; then
  echo "lower I8 constructor does not take loaded ownership before failure" >&2
  exit 1
fi

prepare_body=$(sed -n '/^fn load_exact_numeric_weights(/,/^pub fn prepare(/p' "$pkg/prepare.mbt")
for required in \
  'commit_owner_transition' \
  'OwnerSlotsEmpty' \
  'RawNumericOwner' \
  'AuthenticatedNumericOwner' \
  'LoadedNumericOwner' \
  'ExecutorTransferOwner' \
  'ExecutorConstructionOwner' \
  'ReadyExecutorOwner' \
  'FailedNumericOwner' \
  'FailedExecutorOwner'; do
  if ! printf '%s\n' "$prepare_body" | rg -q "$required"; then
    echo "i8 preparation does not exercise owner transition $required" >&2
    exit 1
  fi
done

lifecycle_test="$pkg/lifecycle_wbtest.mbt"
for required in \
  'i8_resources_publishable' \
  'run_i8_cleanup_chain' \
  'fake_owner_advance' \
  'validate_owner_transition' \
  'PagedCleanupRequired' \
  'post-transfer failure successful cleanup closes lower owner exactly once' \
  'assert_eq\(close_count.val, 1\)' \
  'assert_eq\(active.val, 1\)' \
  'assert_eq\(active.val, 0\)'; do
  if ! rg -n "$required" "$lifecycle_test" >/dev/null; then
    echo "i8 lifecycle tests lack effective anchor $required" >&2
    exit 1
  fi
done

if ! rg -n 'DenseLlamaI8PagedAotV6' "$bootstrap/prepare.mbt" >/dev/null ||
  ! rg -n 'i8_model_source\(\).*None|i8_execution\(\).*None' "$bootstrap/derive_i8.mbt" >/dev/null; then
  echo "bootstrap does not fail closed on the typed I8 recipe" >&2
  exit 1
fi

if ! rg -n 'bootstrap_source_capacity\(\)' "$child" >/dev/null ||
  rg -n 'decode_i8_bootstrap_source_v6|EncodedI8BootstrapSource' "$child" >/dev/null; then
  echo "child bypasses common canonical source dispatch" >&2
  exit 1
fi

release_signature=$(sed -n '/^pub fn prepare_owned(/,/raise I8ReleaseWorkerError {/p' "$release")
for required in \
  'I8RuntimeInstanceAdmission' \
  'WorkerExecutableAdmission' \
  'model_root : @approved_fs.ApprovedRoot' \
  'kernel_root : @approved_fs.ApprovedRoot'; do
  if ! printf '%s\n' "$release_signature" | rg -q "$required"; then
    echo "release prepare signature is missing opaque input $required" >&2
    exit 1
  fi
done
if printf '%s\n' "$release_signature" |
  rg -q 'SchedulerBlueprint|WorkerServiceBinding|Bytes|WorkerProcessLimits|WorkerRestartBackoffPolicy'; then
  echo "release prepare signature permits a caller-substitutable raw input" >&2
  exit 1
fi

expected_release_mbti='pub fn prepare_owned(@instance_admission.I8RuntimeInstanceAdmission, @worker_executable_file.WorkerExecutableAdmission, @approved_fs.ApprovedRoot, @approved_fs.ApprovedRoot) -> @worker_service.OwnedWorkerServicePreparation raise I8ReleaseWorkerError'
if ! rg -F -x "$expected_release_mbti" "$release_mbti" >/dev/null; then
  echo "release MBTI does not expose the exact opaque prepare API (or is stale)" >&2
  exit 1
fi
expected_online_mbti='pub fn prepare_owned_online_framed(@instance_admission.I8RuntimeInstanceAdmission, @worker_executable_file.WorkerExecutableAdmission, @approved_fs.ApprovedRoot, @approved_fs.ApprovedRoot) -> @online_session.LunaOnlineFramedServicePreparation raise I8ReleaseWorkerError'
if ! rg -F -x "$expected_online_mbti" "$release_mbti" >/dev/null; then
  echo "release MBTI does not expose the exact opaque online API (or is stale)" >&2
  exit 1
fi

release_body=$(sed -n '/^pub fn prepare_owned(/,/^}/p' "$release")
for required in \
  'retain_exact_release_inputs' \
  'instance\.scheduler_blueprint\(\)' \
  'retained\.executable\.activation_path\(\)' \
  'instance\.worker_binding\(\)' \
  'instance\.worker_process_limits\(\)' \
  'instance\.restart_policy\(\)' \
  'runtime\.startup_contract\(\)' \
  'runtime\.bootstrap_source\(\)'; do
  if ! printf '%s\n' "$release_body" | rg -q "$required"; then
    echo "release prepare does not forward retained admission evidence: $required" >&2
    exit 1
  fi
done

online_signature=$(sed -n '/^pub fn prepare_owned_online_framed(/,/raise I8ReleaseWorkerError {/p' "$release")
if printf '%s\n' "$online_signature" |
  rg -q 'SchedulerBlueprint|WorkerServiceBinding|Bytes|WorkerProcessLimits|WorkerRestartBackoffPolicy|TokenizerSpec|InferenceLimits|FramedWireLimits'; then
  echo "online release signature permits a caller-substitutable raw input" >&2
  exit 1
fi
online_body=$(sed -n '/^pub fn prepare_owned_online_framed(/,/^}/p' "$release")
for required in \
  'validate_release_join\(instance\)' \
  'instance\.tokenizer\(\)' \
  'instance\.tokenizer_digest\(\)' \
  'runtime\.model_identity\(\)' \
  'instance\.worker_binding\(\)\.inference_limits\(\)' \
  'instance\.framed_limits\(\)' \
  'instance\.preparation_lane_count\(\)' \
  'instance\.preparation_step_budget\(\)' \
  'instance\.preparation_work_limit\(\)' \
  'instance\.preparation_storage_budget\(\)' \
  'instance\.event_step_budget\(\)' \
  'retained\.blueprint' \
  'retained\.executable\.activation_path\(\)' \
  'runtime\.startup_contract\(\)' \
  'runtime\.bootstrap_source\(\)' \
  'instance\.worker_process_limits\(\)' \
  'instance\.restart_policy\(\)' \
  '@online_session\.prepare_owned_luna_online_framed_service'; do
  if ! printf '%s\n' "$online_body" | rg -q "$required"; then
    echo "online release does not derive exact instance input $required" >&2
    exit 1
  fi
done

for required in \
  'source\.digest\(\).*startup\.bootstrap_source_digest' \
  'inspection\.artifact_digest\(\).*execution\.weight_artifact_digest' \
  '@worker_service\.prepare_owned'; do
  if ! rg -n -U "$required" "$release" >/dev/null; then
    echo "release join is missing $required" >&2
    exit 1
  fi
done

for required in \
  'retain_exact_release_inputs' \
  'physical_equal\(retained\.blueprint, blueprint\)' \
  'retained\.executable\.activation_path\(\)' \
  'first\.activation_path\(\)' \
  'second\.activation_path\(\)' \
  'assert_false'; do
  if ! rg -n "$required" "$release_test" >/dev/null; then
    echo "release forwarding test lacks effective anchor $required" >&2
    exit 1
  fi
done

if ! rg -n '^pub fn admit\(@descriptor_file\.RuntimeDescriptorAdmission, @instance_policy_file\.InstancePolicyAdmission, @json_file\.TokenizerFileAdmission\) -> RuntimeInstanceAdmission ' "$instance_mbti" >/dev/null ||
  ! rg -n '^pub fn admit_i8\(@descriptor_file\.I8RuntimeDescriptorAdmission, @instance_policy_file\.InstancePolicyAdmission, @json_file\.TokenizerFileAdmission\) -> I8RuntimeInstanceAdmission ' "$instance_mbti" >/dev/null ||
  ! rg -n '^pub struct I8RuntimeInstanceAdmission \{' "$instance_mbti" >/dev/null ||
  ! sed -n '/^pub struct I8RuntimeInstanceAdmission {/,/^}/p' "$instance_mbti" | rg -q 'private fields'; then
  echo "instance admission MBTI lacks parallel opaque I8 factory (or is stale)" >&2
  exit 1
fi

for required in \
  'require_runtime_digest' \
  'require_layout_geometry' \
  'require_tokenizer_identity' \
  'require_tokenizer_envelope' \
  'require_blueprint_capacity' \
  'instance coherence substitution was admitted'; do
  if ! rg -n "$required" "$instance/coherence_wbtest.mbt" >/dev/null; then
    echo "instance hostile test lacks effective anchor $required" >&2
    exit 1
  fi
done

cleanup_body=$(sed -n '/^fn close_paged_resources(/,/^}/p' "$paged_cleanup")
if ! printf '%s\n' "$cleanup_body" | rg -q 'run_paged_weight_dependency_close'; then
  echo "paged cleanup bypasses the lease-before-numeric dependency gate" >&2
  exit 1
fi
dependency_body=$(sed -n '/^fn run_paged_weight_dependency_close(/,/^}/p' "$paged_cleanup")
if ! printf '%s\n' "$dependency_body" | rg -q 'close_lease\(\)' ||
  ! printf '%s\n' "$dependency_body" | rg -q 'if !lease_open\(\)' ||
  ! printf '%s\n' "$dependency_body" | rg -q 'close_numeric\(\)'; then
  echo "paged weight dependency gate is structurally incomplete" >&2
  exit 1
fi

echo "i8 worker and release boundaries: ok"
