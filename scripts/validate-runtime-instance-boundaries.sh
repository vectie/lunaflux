#!/usr/bin/env bash
set -eu

failed=0
fail() {
  printf '%s\n' "$1" >&2
  failed=1
}

if rg -n 'lunanexa|moongate|runtime JIT|compiler invocation|global mutable' \
  deploy/launch_file deploy/worker_executable_file ops/runtime_instance \
  cmd/lunaflux/native_run.mbt --glob '*.mbt' --glob 'moon.pkg'; then
  fail 'operator runtime boundary contains a forbidden product/JIT/global edge'
fi

if [ "$(rg -c '"lunaflux\.launch\.json"' deploy/launch_file/load.mbt)" -ne 1 ] ||
  ! rg -q 'let marker = "#sha256="' deploy/launch_file/load.mbt ||
  ! rg -q 'file\.read_immutable_snapshot' deploy/launch_file/load.mbt ||
  ! rg -q 'let root_close.*root\.close' deploy/launch_file/load.mbt; then
  fail 'launch admission is not one fixed independently pinned closed snapshot'
fi

if ! rg -q 'maximum_bytes > 1048576L' deploy/launch_file/limits.mbt ||
  ! rg -q 'maximum_json_depth > 16' deploy/launch_file/limits.mbt ||
  ! rg -q 'maximum_bytes > 1073741824L' \
    deploy/worker_executable_file/admit.mbt; then
  fail 'launch or executable startup bounds lost their absolute ceilings'
fi

if rg -n '\.require_absolute_identity\(' \
    deploy/worker_executable_file/admit.mbt ||
  [ "$(rg -c 'raw_open_executable\(' \
    deploy/worker_executable_file/admit.mbt)" -ne 1 ] ||
  [ "$(rg -c 'raw_snapshot_and_pin\(' \
    deploy/worker_executable_file/admit.mbt)" -ne 1 ] ||
  ! rg -Fq 'ignore(split_absolute_path(absolute_path))' \
    deploy/worker_executable_file/admit.mbt ||
  ! rg -q '@crypto\.sha256' deploy/worker_executable_file/admit.mbt ||
  ! rg -Fq \
    '{ activation_path: @utf8.encode(absolute_path), digest: expected, handle }' \
    deploy/worker_executable_file/admit.mbt; then
  fail 'worker executable adapter lost direct no-follow snapshot/digest ownership'
fi

for required in \
  '@launch_file.load_argument' \
  '@descriptor_file.load' \
  '@descriptor_file.load_i8_v2' \
  '@instance_policy_file.load' \
  '@json_file.load' \
  '@instance_admission.admit' \
  '@instance_admission.admit_i8' \
  '@hardware.probe' \
  '@worker_executable_file.verify' \
  '@online_session.prepare_owned_luna_online_framed_service' \
  '@i8_release_worker.prepare_owned_online_framed' \
  '@online_tcp.bind_luna_online_tcp_pipeline_server'; do
  if ! rg -q "$required" ops/runtime_instance; then
    fail "runtime owner composition lost required existing boundary: $required"
  fi
done

physical_line=$(rg -n 'admitted\.runtime\.admit_physical\(\)' \
  ops/runtime_instance/prepare.mbt | cut -d: -f1)
executable_line=$(rg -n 'verify_release_worker\(admitted\.launch' \
  ops/runtime_instance/prepare.mbt | cut -d: -f1)
service_line=$(rg -n 'admitted\.instance\.prepare_owned_service' \
  ops/runtime_instance/prepare.mbt | cut -d: -f1)
if [ -z "$physical_line" ] || [ -z "$executable_line" ] ||
  [ -z "$service_line" ] || [ "$physical_line" -ge "$executable_line" ] ||
  [ "$executable_line" -ge "$service_line" ]; then
  fail 'physical/executable/service startup ordering is not fail-closed'
fi

if [ "$(rg -c '@descriptor_file\.load\(' \
  ops/runtime_instance/runtime_route.mbt)" -ne 1 ] ||
  [ "$(rg -c '@descriptor_file\.load_i8_v2\(' \
    ops/runtime_instance/runtime_route.mbt)" -ne 1 ] ||
  [ "$(rg -c '@instance_admission\.admit\(' \
    ops/runtime_instance/runtime_route.mbt)" -ne 1 ] ||
  [ "$(rg -c '@instance_admission\.admit_i8\(' \
    ops/runtime_instance/runtime_route.mbt)" -ne 1 ] ||
  ! rg -Fq 'DenseLlamaPagedAotV5 => legacy_loader()' \
    ops/runtime_instance/runtime_route.mbt ||
  ! rg -Fq 'DenseLlamaI8PagedAotV6 => i8_loader()' \
    ops/runtime_instance/runtime_route.mbt; then
  fail 'authenticated runtime recipe can probe, retry, or cross-decode loaders'
fi

if ! rg -Fq 'priv enum AdmittedRuntimeDescriptor {' \
  ops/runtime_instance/runtime_route_types.mbt ||
  ! rg -Fq 'priv enum AdmittedRuntimeInstance {' \
    ops/runtime_instance/runtime_route_types.mbt ||
  ! rg -Fq 'priv struct RuntimeServiceEnvelope {' \
    ops/runtime_instance/runtime_route_types.mbt ||
  rg -q 'pub.*(AdmittedRuntimeDescriptor|AdmittedRuntimeInstance|RuntimeServiceEnvelope)|LegacyRuntimeDescriptor|I8RuntimeDescriptor|LegacyRuntimeInstance|I8RuntimeInstance' \
    ops/runtime_instance/pkg.generated.mbti ||
  ! rg -q 'pub fn prepare\(StringView, RuntimeInstanceFileLimits\) -> RuntimeInstanceOwner raise RuntimeInstanceError' \
    ops/runtime_instance/pkg.generated.mbti; then
  fail 'runtime recipe discrimination leaked or public one-argument preparation drifted'
fi

if ! rg -Fq 'pub struct RuntimeBoundEndpoint {' \
    ops/runtime_instance/pkg.generated.mbti ||
  ! rg -Fq \
    'pub fn RuntimeInstanceOwner::bound_endpoint(Self) -> RuntimeBoundEndpoint raise RuntimeInstanceError' \
    ops/runtime_instance/pkg.generated.mbti ||
  ! rg -Fq \
    'pub fn RuntimeInstanceOwner::control_origin(Self) -> String raise RuntimeInstanceError' \
    ops/runtime_instance/pkg.generated.mbti ||
  rg -n 'RuntimeBoundEndpoint.*(@socket|Control|Root|Credential)' \
    ops/runtime_instance/pkg.generated.mbti ||
  [ "$(rg -c 'println\(runtime_origin_line\(endpoint\.origin\(\)\)\)' \
    cmd/lunaflux/native_run.mbt)" -ne 1 ] ||
  [ "$(rg -c 'println\(control_origin_line\(owner\.control_origin\(\)\)\)' \
    cmd/lunaflux/native_run.mbt)" -ne 1 ] ||
  rg -n 'operational_local_addr\(' cmd/lunaflux/native_run.mbt; then
  fail 'ready serving/control origins are not exact root-free publications'
fi

envelope_line=$(rg -n 'admitted\.instance\.service_envelope\(\)' \
  ops/runtime_instance/prepare.mbt | cut -d: -f1)
capture_line=$(rg -n '^  capture_prepared_owner\(' \
  ops/runtime_instance/prepare.mbt | cut -d: -f1)
if [ -z "$envelope_line" ] || [ -z "$capture_line" ] ||
  [ "$envelope_line" -ge "$service_line" ] ||
  [ "$service_line" -ge "$capture_line" ] ||
  ! rg -q 'addr: envelope\.addr' ops/runtime_instance/prepare_helpers.mbt ||
  ! rg -q 'transport: envelope\.transport' \
    ops/runtime_instance/prepare_helpers.mbt ||
  ! rg -q 'telemetry_capacity: envelope\.telemetry_capacity' \
    ops/runtime_instance/prepare_helpers.mbt; then
  fail 'service/address/transport/telemetry do not converge from one instance'
fi

if ! rg -q 'pub fn preflight_release\(' \
  ops/runtime_instance/release_preflight.mbt ||
  ! rg -q 'let admitted = admit_release_inputs\(' \
    ops/runtime_instance/release_preflight.mbt ||
  ! rg -q 'close_model_kernel\(admitted\.model_root, admitted\.kernel_root\)' \
    ops/runtime_instance/release_preflight.mbt ||
  rg -n '@hardware\.probe|admit_physical|prepare_owned_service|@online_tcp\.|@online_session\.' \
    ops/runtime_instance/release_preflight.mbt \
    ops/runtime_instance/release_inputs.mbt ||
  ! rg -q '@runtime_instance\.preflight_release\(' \
    cmd/lunaflux/native_run.mbt; then
  fail 'offline release preflight gained device/process/listener authority or lost cleanup'
fi

for required in \
  '@descriptor_file.load_materialized' \
  '@descriptor_file.load_i8_v2_materialized' \
  '@descriptor_file.load_mistral_v1_materialized' \
  '@descriptor_file.load_tensor_parallel_materialized' \
  '@descriptor_file.load_fp8_reusable_v3_materialized' \
  '@instance_policy_file.load_materialized' \
  '@json_file.load_materialized' \
  '@worker_executable_file.verify_materialized'; do
  if ! rg -q "$required" ops/runtime_instance; then
    fail "materialized release join lost typed mapped admission: $required"
  fi
done
if ! rg -q 'pub fn preflight_materialized_release\(' \
    ops/runtime_instance/materialized_release_preflight.mbt ||
  ! rg -q 'let bundle = load_materialized_bundle' \
    ops/runtime_instance/materialized_release_preflight.mbt ||
  ! rg -Fq 'close_three_roots(' \
    ops/runtime_instance/materialized_release_preflight.mbt ||
  rg -n '@hardware\.probe|admit_physical|prepare_owned_service|@online_tcp\.|@online_session\.' \
    ops/runtime_instance/materialized_*.mbt ||
  ! rg -q '@runtime_instance\.preflight_materialized_release\(' \
    cmd/lunaflux/native_run.mbt; then
  fail 'materialized preflight gained device/process/listener authority or lost cleanup'
fi
if [ "$(rg -c '\.materialization_view\(' \
    ops/runtime_instance/materialized_release_inputs.mbt)" -ne 1 ] ||
  ! rg -q 'fn open_materialization_view\(' \
    ops/runtime_instance/materialized_release_inputs.mbt ||
  ! rg -q 'source_target_binding=1' \
    ops/runtime_instance/materialized_release_preflight.mbt ||
  ! rg -q 'filesystem_authority_closed=1' \
    ops/runtime_instance/materialized_release_preflight.mbt ||
  ! rg -q 'tensor_parallel_group_template_sha256=' \
    ops/runtime_instance/materialized_release_preflight.mbt ||
  rg -q '0000000000000000000000000000000000000000000000000000000000000000' \
    ops/runtime_instance/release_preflight.mbt; then
  fail 'typed source-capability/target-label binding is not centralized and explicit'
fi

if ! rg -Fq \
  'pub fn preflight_release(StringView, RuntimeInstanceFileLimits) -> RuntimeReleaseAdmission raise RuntimeInstanceError' \
  ops/runtime_instance/pkg.generated.mbti ||
  ! rg -Fq 'pub struct RuntimeReleaseAdmission {' \
    ops/runtime_instance/pkg.generated.mbti ||
  [ "$(rg -c '^pub fn RuntimeReleaseAdmission::' \
    ops/runtime_instance/pkg.generated.mbti)" -ne 7 ] ||
  rg -n 'RuntimeReleaseAdmission.*(ApprovedRoot|Device|Process|Socket|Compiler)' \
    ops/runtime_instance/pkg.generated.mbti; then
  fail 'offline release preflight public evidence is not the exact root-free API'
fi

for hostile in \
  'v1 launch never falls back to the I8 descriptor loader' \
  'v2 I8 launch never falls back to the legacy descriptor loader' \
  'missing-hostile-config.json' \
  'missing-hostile-numeric.safetensors' \
  'missing-hostile-policy.json' \
  'cross-recipe descriptor was admitted through fallback' \
  'selected legacy loader failure never probes I8 fallback' \
  'selected I8 loader failure never probes legacy fallback' \
  'legacy_calls.val != expected_legacy_calls' \
  'i8_calls.val != expected_i8_calls'; do
  if ! rg -Fq "$hostile" ops/runtime_instance/runtime_route_wbtest.mbt; then
    fail "runtime cross-recipe hostile evidence lost anchor: $hostile"
  fi
done

if ! rg -q --pcre2 -U \
  'let service = preparation\.take_ready\(\) catch \{(?s:.*?)\}\n  owner\.services\.push\(service\)\n  owner\.record_log\(LunaInstanceReady, LunaLogNoReason\)\n  owner' \
  service/online_tcp/server_prepare.mbt; then
  fail 'binder may fail after consuming preparation or lose retry ownership'
fi

if rg -n --pcre2 -U \
  'RootBoundCleanupRequired\(failed\).*?failed\.is_closed\(\).*?raise' \
  engine/worker_service/owned_prepare.mbt ||
  ! rg -q 'failed_owner\.closed = failed\.is_closed\(\)' \
    engine/worker_service/owned_prepare.mbt ||
  ! rg -q 'RootBoundCleanupRequired\(failed_owner\)' \
    engine/worker_process/root_bound_prepare.mbt; then
  fail 'post-root-acquisition startup failure is not an explicit cleanup outcome'
fi

if ! rg -Fq 'owner.preparations.push(service)' \
  ops/runtime_instance/prepare_helpers.mbt ||
  ! rg -Fq 'LunaOnlineInstanceCleanupRequired => {' \
    ops/runtime_instance/prepare_helpers.mbt ||
  ! rg -Fq 'self.preparations[0].take_cleanup' \
    ops/runtime_instance/owner.mbt; then
  fail 'operator can abandon a temporarily unextractable cleanup preparation'
fi

if ! rg -Fq 'close_or_retain_borrowed_root(owner, kernel_root)' \
  ops/runtime_instance/prepare_helpers.mbt ||
  ! rg -Fq 'close_or_retain_borrowed_root(owner, model_root)' \
    ops/runtime_instance/prepare_helpers.mbt ||
  ! rg -Fq 'borrowed_roots.pop' ops/runtime_instance/owner.mbt ||
  ! rg -Fq 'self.borrowed_roots.length() == 0' \
    ops/runtime_instance/owner.mbt ||
  ! rg -q 'borrowed root close outcomes retain only live failed authority' \
    ops/runtime_instance/owner_wbtest.mbt ||
  ! rg -q 'captured lower outcome closes both borrowed originals exactly once' \
    ops/runtime_instance/owner_wbtest.mbt ||
  ! rg -q 'failed owner retries only a retained live borrowed original' \
    ops/runtime_instance/owner_wbtest.mbt; then
  fail 'borrowed startup roots can be double-closed or lost after transfer'
fi

if rg -n 'try! .*begin_drain' ops/runtime_instance cmd/lunaflux/native_run.mbt ||
  ! rg -q 'drain_native_owner' cmd/lunaflux/native_run.mbt ||
  ! rg -q 'native cleanup retries a failed first drain request' \
    cmd/lunaflux/main_wbtest.mbt ||
  ! rg -q 'owner\.cleanup_complete\(\)' cmd/lunaflux/native_run.mbt ||
  ! rg -q 'Starting | HealthyNotReady | Ready' cmd/lunaflux/native_run.mbt; then
  fail 'operator lifecycle can abort or exit before deterministic cleanup'
fi

if ! rg -q 'phase == Ready && listener_bound' \
    ops/runtime_instance/owner_status.mbt ||
  ! rg -q 'self\.servers\[0\]\.health\(\) == LunaOnlineTcpServerFailed' \
    ops/runtime_instance/owner.mbt ||
  ! rg -q 'self\.servers\[0\]\.begin_drain' ops/runtime_instance/owner.mbt ||
  ! rg -q 'self\.openai_servers\[0\]\.begin_drain' \
    ops/runtime_instance/owner.mbt ||
  ! rg -q 'record_cold_start_latency_millis' \
    ops/runtime_instance/ingress_owner.mbt; then
  fail 'owner readiness, existing-server drain, or cold-start seam drifted'
fi

launch_importers=$(rg -l 'vectie/lunaflux/deploy/launch_file' \
  --glob '**/moon.pkg' . | sed 's#^\./##' |
  sort)
if [ "$launch_importers" != $'cmd/lunaflux/moon.pkg\nops/runtime_instance/moon.pkg\ntests/approved_model_spawned_physical/moon.pkg' ]; then
  fail 'opaque launch evidence has unauthorized consumers'
fi

worker_executable_importer_allowed() {
  case "$1" in
    cmd/lunaflux/moon.pkg | \
      engine/rank_group_process/moon.pkg | \
      engine/tensor_parallel_group_transport/moon.pkg | \
      engine/worker_process/moon.pkg | \
      engine/worker_service/moon.pkg | \
      ops/runtime_instance/moon.pkg | \
      runtime/fp8_release_worker/moon.pkg | \
      runtime/i8_release_worker/moon.pkg | \
      service/online_session/moon.pkg | \
      tests/approved_model_spawned_physical/moon.pkg | \
      tests/worker_executable_fixture/moon.pkg) return 0 ;;
    *) return 1 ;;
  esac
}

# Positive and hostile controls keep this an exact package grant rather than a
# prefix/suffix allowlist that could silently admit a lookalike owner.
worker_executable_importer_allowed engine/worker_process/moon.pkg ||
  fail 'worker executable importer allowlist rejected the process owner'
if worker_executable_importer_allowed \
  engine/worker_process_shadow/moon.pkg; then
  fail 'worker executable importer allowlist accepted a near-name owner'
fi
if worker_executable_importer_allowed \
  engine/worker_process/moon.pkg.injected; then
  fail 'worker executable importer allowlist accepted a suffixed package file'
fi

executable_importers=$(rg -l \
  'vectie/lunaflux/deploy/worker_executable_file' \
  --glob '**/moon.pkg' . | sed 's#^\./##' | sort)
while IFS= read -r executable_importer; do
  [ -z "$executable_importer" ] ||
    worker_executable_importer_allowed "$executable_importer" ||
    fail "opaque worker executable evidence has an unauthorized consumer: $executable_importer"
done <<<"$executable_importers"
expected_executable_importers=$'cmd/lunaflux/moon.pkg\nengine/rank_group_process/moon.pkg\nengine/tensor_parallel_group_transport/moon.pkg\nengine/worker_process/moon.pkg\nengine/worker_service/moon.pkg\nops/runtime_instance/moon.pkg\nruntime/fp8_release_worker/moon.pkg\nruntime/i8_release_worker/moon.pkg\nservice/online_session/moon.pkg\ntests/approved_model_spawned_physical/moon.pkg\ntests/worker_executable_fixture/moon.pkg'
if [ "$executable_importers" != "$expected_executable_importers" ]; then
  fail 'opaque worker executable ownership graph drifted'
fi

live_executable_verifiers=$(rg -n '@worker_executable_file\.verify\(' \
  --glob '*.mbt' --glob '!*_test.mbt' --glob '!*_wbtest.mbt' \
  --glob '!tests/**' . |
  sed 's#^\./##' || true)
if [ "$(printf '%s\n' "$live_executable_verifiers" | sed '/^$/d' |
    wc -l | tr -d ' ')" -ne 1 ] ||
  ! printf '%s\n' "$live_executable_verifiers" |
    rg -q '^ops/runtime_instance/release_preflight\.mbt:'; then
  fail 'live worker executable mint escaped the singular runtime owner'
fi
materialized_executable_verifiers=$(rg -n \
  '@worker_executable_file\.verify_materialized\(' \
  --glob '*.mbt' --glob '!*_test.mbt' --glob '!*_wbtest.mbt' . |
  sed 's#^\./##' || true)
if [ "$(printf '%s\n' "$materialized_executable_verifiers" | sed '/^$/d' |
    wc -l | tr -d ' ')" -ne 1 ] ||
  ! printf '%s\n' "$materialized_executable_verifiers" |
    rg -q '^ops/runtime_instance/materialized_release_inputs\.mbt:'; then
  fail 'materialized worker evidence verification escaped its offline join'
fi
child_preparation_calls=$(rg -n 'executable\.prepare_child\(' \
  engine --glob '*.mbt' --glob '!*_test.mbt' --glob '!*_wbtest.mbt' || true)
if [ "$(printf '%s\n' "$child_preparation_calls" | sed '/^$/d' |
    wc -l | tr -d ' ')" -ne 3 ] ||
  [ "$(printf '%s\n' "$child_preparation_calls" |
    rg -c '^engine/rank_group_process/prepare\.mbt:')" -ne 2 ] ||
  ! printf '%s\n' "$child_preparation_calls" |
    rg -q '^engine/worker_process/startup_preflight\.mbt:' ||
  ! printf '%s\n' "$child_preparation_calls" |
    rg -q '^engine/rank_group_process/prepare\.mbt:'; then
  fail 'pinned executable activation escaped the two process preparation owners'
fi
if ! rg -U -q \
  'pub struct WorkerExecutableAdmission \{\n  // private fields\n\}' \
  deploy/worker_executable_file/pkg.generated.mbti ||
  rg -n 'ApprovedExecutableHandle|@cap\.' \
    deploy/worker_executable_file/pkg.generated.mbti; then
  fail 'worker executable admission is no longer opaque at the public boundary'
fi

if ! rg -q '"lunaflux\.launch\.v1"' deploy/launch_file/schema.mbt ||
  ! rg -q '"lunaflux\.launch\.v2"' deploy/launch_file/schema.mbt ||
  ! rg -q 'DenseLlamaPagedAotV5' deploy/launch_file/schema.mbt ||
  ! rg -q 'DenseLlamaI8PagedAotV6' deploy/launch_file/schema.mbt ||
  ! rg -q 'runtime_recipe == DenseLlamaI8PagedAotV6.*luna_approval is Some' \
    deploy/launch_file/schema.mbt ||
  ! rg -q 'launch v1 v2 recipe and approval cross decoding fails closed' \
    deploy/launch_file/schema_wbtest.mbt; then
  fail 'launch v1/v2 recipe or incompatible approval admission drifted'
fi

launch_recipe_interface=$(sed -n \
  '/^pub(all) enum LaunchRuntimeRecipe {$/,/^} derive(Eq, @debug.Debug)$/p' \
  deploy/launch_file/pkg.generated.mbti)
expected_launch_recipe_interface='pub(all) enum LaunchRuntimeRecipe {
  DenseLlamaPagedAotV5
  DenseLlamaI8PagedAotV6
  DenseMistralBf16PagedAotV7
  DenseLlamaTensorParallelPagedAotApprovedV8
  DenseLlamaFp8ReusablePagedAotApprovedV9
} derive(Eq, @debug.Debug)'
if [ "$launch_recipe_interface" != "$expected_launch_recipe_interface" ] ||
  ! rg -Fq 'pub fn LaunchFileAdmission::runtime_recipe(Self) -> LaunchRuntimeRecipe' \
    deploy/launch_file/pkg.generated.mbti; then
  fail 'generated launch recipe interface is not the exact closed allowlist'
fi

if ! rg -q '@runtime_instance\.prepare_opaque_cli\(' \
  cmd/lunaflux/native_run.mbt ||
  rg -n '@descriptor_file\.(load|load_i8_v2)|@instance_admission\.(admit|admit_i8)' \
    cmd/lunaflux/native_run.mbt; then
  fail 'native one-argument run no longer reaches the opaque runtime owner'
fi

while IFS= read -r file; do
  lines=$(wc -l < "$file" | tr -d ' ')
  if [ "$lines" -ge 500 ]; then
    fail "operator runtime source exceeds file budget: $file ($lines)"
  fi
done < <(find deploy/launch_file deploy/worker_executable_file \
  ops/runtime_instance -name '*.mbt' -type f | sort)

if [ "$failed" -ne 0 ]; then
  exit 1
fi

printf '%s\n' 'runtime instance launch/authority boundary gate passed'
