#!/usr/bin/env bash

set -euo pipefail
LC_ALL=C
TZ=UTC
export LC_ALL TZ
umask 077

fail() {
  printf '%s\n' "LunaFlux current-source physical campaign rejected: $1" >&2
  exit 1
}

usage() {
  printf '%s\n' \
    'usage: run-approved-model-current-source-physical.sh ABSOLUTE_NVCC_13_1 ABSOLUTE_NEW_EVIDENCE_DIR' >&2
  exit 2
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  else
    shasum -a 256 "$1" | cut -d ' ' -f 1
  fi
}

verify_canonical_evidence() {
  local evidence_file=$1
  shift
  local digest_key=$1
  shift
  local expected_nonempty_lines=$1
  shift
  local label=$1
  local digest digest_count last_nonempty nonempty_lines payload calculated
  digest=$(sed -n "s/^${digest_key}=//p" "$evidence_file")
  digest_count=$(grep -c "^${digest_key}=" "$evidence_file" || true)
  [ "$digest_count" -eq 1 ] || fail "$label published duplicate digest authority"
  case "$digest" in ''|*[!0-9a-f]*) fail "$label published an invalid digest" ;; esac
  [ "${#digest}" -eq 64 ] || fail "$label published an invalid digest"
  last_nonempty=$(grep -v '^$' "$evidence_file" | tail -n 1)
  [ "$last_nonempty" = "$digest_key=$digest" ] ||
    fail "$label published data after its canonical digest"
  nonempty_lines=$(grep -c . "$evidence_file" || true)
  [ "$nonempty_lines" -eq "$expected_nonempty_lines" ] ||
    fail "$label evidence shape does not match its source contract"
  payload=$(mktemp) || fail "$label could not allocate digest scratch space"
  scratch_files+=("$payload")
  sed "/^${digest_key}=/,\$d" "$evidence_file" >"$payload" ||
    fail "$label canonical payload extraction failed"
  calculated=$(sha256_file "$payload") || fail "$label payload hashing failed"
  [ "$calculated" = "$digest" ] || fail "$label canonical digest does not verify"
}

make_sorted_path_list() {
  local destination=$1
  shift
  local unsorted
  unsorted=$(mktemp) || fail 'could not allocate a path-list scratch file'
  scratch_files+=("$unsorted")
  find "$@" -print0 >"$unsorted" || fail 'source path discovery failed'
  LC_ALL=C sort -z "$unsorted" >"$destination" ||
    fail 'source path ordering failed'
}

[ "$#" -eq 2 ] || usage
nvcc=$1
evidence_dir=$2
case "$nvcc" in /*) ;; *) usage ;; esac
case "$evidence_dir" in /*) ;; *) usage ;; esac
[ -x "$nvcc" ] && [ ! -L "$nvcc" ] || usage
[ "$(realpath -- "$nvcc")" = "$nvcc" ] || usage
[ ! -e "$evidence_dir" ] && [ ! -L "$evidence_dir" ] ||
  fail 'evidence directory already exists'
evidence_parent=$(CDPATH= cd -- "$(dirname -- "$evidence_dir")" && pwd -P)
[ "$evidence_parent/$(basename -- "$evidence_dir")" = "$evidence_dir" ] ||
  fail 'evidence directory path is not canonical'

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/immutable-evidence-directory.sh"
. "$repo_root/scripts/physical-campaign-process-group.sh"
case "$evidence_dir" in "$repo_root"|"$repo_root"/*)
  fail 'evidence directory must be outside the source tree'
esac
appledouble=$(find "$repo_root" \
  \( -path "$repo_root/.git" -o -path "$repo_root/_build" -o \
    -path "$repo_root/trace.json" \) -prune -o \
  -name '._*' -print | sed -n '1p')
[ -z "$appledouble" ] || fail "AppleDouble source entry exists: $appledouble"
special=$(find "$repo_root" \
  \( -path "$repo_root/.git" -o -path "$repo_root/_build" -o \
    -path "$repo_root/trace.json" \) -prune -o \
  ! -type d ! -type f ! -type l -print | sed -n '1p')
[ -z "$special" ] || fail "special source entry exists: $special"

mkdir "$evidence_dir"
work=$evidence_dir/work
logs=$evidence_dir/logs
mkdir "$work" "$logs"
stage=source-inventory
campaign_complete=0
source_files_sha=unavailable
source_links_sha=unavailable
worker_sha=unavailable
launch_sha=unavailable
prefix_launch_sha=unavailable
host_referee_launch_sha=unavailable
scratch_files=()

finalize_campaign() {
  status=$?
  trap - EXIT
  trap '' HUP INT TERM
  if ! lunaflux_stop_campaign_group; then
    status=1
    stage=process-group-cleanup-failed
    {
      printf '%s\n' 'evidence_schema=lunaflux.approved-model.current-source-physical.v1'
      printf '%s\n' 'outcome=failed'
      printf '%s\n' 'exit_status=1'
      printf '%s\n' 'terminal_stage=process-group-cleanup-failed'
      printf 'surviving_process_group=%s\n' "$lunaflux_campaign_pgid"
      printf '%s\n' 'evidence_sealed=0'
      printf '%s\n' 'files_manifest_sha256=unavailable'
      printf '%s\n' 'compiler_authority=offline-only'
      printf '%s\n' 'request_path_jit=0'
    } >"$evidence_dir/RESULT.txt"
    chmod 0444 "$evidence_dir/RESULT.txt" 2>/dev/null || true
    exit "$status"
  fi
  for scratch in "${scratch_files[@]}"; do
    rm -f -- "$scratch"
  done
  lunaflux_prepare_evidence_manifest "$evidence_dir" || exit 1
  files_sha=$lunaflux_evidence_manifest_sha256
  outcome=failed
  if [ "$campaign_complete" -eq 1 ] && [ "$status" -eq 0 ]; then
    outcome=passed
  fi
  {
    printf '%s\n' 'evidence_schema=lunaflux.approved-model.current-source-physical.v1'
    printf 'outcome=%s\n' "$outcome"
    printf 'exit_status=%s\n' "$status"
    printf 'terminal_stage=%s\n' "$stage"
    printf 'source_files_sha256=%s\n' "$source_files_sha"
    printf 'source_links_sha256=%s\n' "$source_links_sha"
    printf 'worker_executable_sha256=%s\n' "$worker_sha"
    printf 'launch_sha256=%s\n' "$launch_sha"
    printf 'prefix_launch_sha256=%s\n' "$prefix_launch_sha"
    printf 'host_referee_launch_sha256=%s\n' "$host_referee_launch_sha"
    printf 'files_manifest_sha256=%s\n' "$files_sha"
    printf '%s\n' 'compiler_authority=offline-only'
    printf '%s\n' 'request_path_jit=0'
  } >"$evidence_dir/RESULT.txt"
  if [ -f "$work/launch/bin/lunaflux-device-worker" ]; then
    lunaflux_seal_evidence_directory \
      "$evidence_dir" "$work/launch/bin/lunaflux-device-worker" || exit 1
  else
    lunaflux_seal_evidence_directory "$evidence_dir" || exit 1
  fi
  exit "$status"
}
trap finalize_campaign EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

write_source_inventory() {
  local files_output=$1
  local links_output=$2
  local file relative link target digest file_paths link_paths
  file_paths=$(mktemp) || fail 'could not allocate source-file scratch space'
  link_paths=$(mktemp) || fail 'could not allocate source-link scratch space'
  scratch_files+=("$file_paths" "$link_paths")
  make_sorted_path_list "$file_paths" "$repo_root" \
    \( -path "$repo_root/.git" -o -path "$repo_root/_build" -o \
      -path "$repo_root/trace.json" \) -prune -o \
    -type f ! -name '.DS_Store'
  make_sorted_path_list "$link_paths" "$repo_root" \
    \( -path "$repo_root/.git" -o -path "$repo_root/_build" -o \
      -path "$repo_root/trace.json" \) -prune -o -type l
  : >"$files_output"
  while IFS= read -r -d '' file; do
    relative=${file#"$repo_root"/}
    case "$relative" in *$'\n'*) fail 'source filename contains a newline' ;; esac
    digest=$(sha256_file "$file") || fail "could not hash source file: $relative"
    printf '%s  %s\n' "$digest" "$relative" >>"$files_output"
  done <"$file_paths"
  : >"$links_output"
  while IFS= read -r -d '' link; do
    relative=${link#"$repo_root"/}
    target=$(readlink -- "$link")
    case "$relative$target" in *$'\n'*) fail 'source symlink contains a newline' ;; esac
    printf '%s -> %s\n' "$relative" "$target" >>"$links_output"
  done <"$link_paths"
}

source_files=$evidence_dir/source.files.sha256
source_links=$evidence_dir/source.symlinks.v1
write_source_inventory "$source_files" "$source_links"
source_files_sha=$(sha256_file "$source_files")
source_links_sha=$(sha256_file "$source_links")

cd "$repo_root"
command -v setsid >/dev/null 2>&1 ||
  fail 'setsid is required for deterministic campaign cleanup'
stage=static-boundaries
scripts/validate-approved-bf16-model-physical.sh \
  >"$logs/validate-approved-bf16.stdout" \
  2>"$logs/validate-approved-bf16.stderr"
scripts/validate-approved-model-spawned-physical.sh \
  >"$logs/validate-spawned.stdout" 2>"$logs/validate-spawned.stderr"
scripts/validate-approved-model-context-churn.sh \
  >"$logs/validate-context-churn.stdout" \
  2>"$logs/validate-context-churn.stderr"
scripts/validate-approved-model-long-context.sh \
  >"$logs/validate-long-context.stdout" \
  2>"$logs/validate-long-context.stderr"
scripts/validate-approved-model-serving-readiness.sh \
  >"$logs/validate-serving.stdout" 2>"$logs/validate-serving.stderr"
scripts/validate-approved-model-broad-bf16-serving.sh \
  >"$logs/validate-broad-bf16-serving.stdout" \
  2>"$logs/validate-broad-bf16-serving.stderr"
scripts/validate-online-listener-restart-boundary.sh \
  >"$logs/validate-listener-restart.stdout" \
  2>"$logs/validate-listener-restart.stderr"
scripts/validate-openai-qualification-ownership.sh \
  >"$logs/validate-openai-qualification.stdout" \
  2>"$logs/validate-openai-qualification.stderr"

stage=toolchain-identity
scripts/inspect-luna-cuda-aot-driver.sh "$nvcc" \
  >"$work/driver.v1" 2>"$logs/driver.stderr" ||
  fail 'CUDA toolchain identity inspection failed'
[ ! -s "$logs/driver.stderr" ] || fail 'CUDA toolchain inspection emitted stderr'
compiler_version=$(sed -n '7s/^compiler_version=//p' "$work/driver.v1")
[ "$compiler_version" = 13.1.115 ] || fail 'CUDA compiler is not exact 13.1.115'
driver_identity=$(sed -n '6s/^driver_identity_sha256=//p' "$work/driver.v1")
compiler_major=${compiler_version%%.*}
compiler_tail=${compiler_version#*.}
compiler_minor=${compiler_tail%%.*}
compiler_patch=${compiler_tail#*.}
toolchain_manifest=$work/toolchain.v1
{
  printf '%s\n' 'schema=lunaflux-approved-complete-cuda-toolchain-v1'
  printf 'driver_identity_sha256=%s\n' "$driver_identity"
} >"$toolchain_manifest"
toolchain_sha=$(sha256_file "$toolchain_manifest")

stage=build-candidate-exporter
moon build --target native --deny-warn tests/approved_bf16_model_physical \
  >"$logs/build-exporter.stdout" 2>"$logs/build-exporter.stderr"
exporter=$repo_root/_build/native/debug/build/tests/approved_bf16_model_physical/approved_bf16_model_physical.exe
[ -x "$exporter" ] || fail 'approved-model candidate exporter is missing'

stage=export-candidates
model_root=$repo_root/tests/reference_corpus
export_root=$work/export
"$exporter" export "$model_root" "$toolchain_sha" \
  "$compiler_major" "$compiler_minor" "$compiler_patch" "$export_root" \
  >"$logs/export.stdout" 2>"$logs/export.stderr"
[ ! -s "$logs/export.stderr" ] || fail 'candidate export emitted stderr'
grep -Eq '^outcome=approved-bf16-candidates-exported target=sm_120 operations=21 candidate_set_sha256=[0-9a-f]{64} inventory_sha256=[0-9a-f]{64} export_record_sha256=[0-9a-f]{64}$' \
  "$logs/export.stdout" || fail 'candidate export evidence is invalid'

stage=compile-aot-set
candidate_root=$export_root/candidate-root
candidate_inventory=$export_root/candidate.files.sha256
candidate_inventory_sha=$(sha256_file "$candidate_inventory")
compiled_root=$work/compiled
scripts/build-luna-bf16-kernel-set.sh \
  "$nvcc" "$toolchain_manifest#sha256=$toolchain_sha" \
  "$candidate_root" "$candidate_inventory#sha256=$candidate_inventory_sha" \
  "$compiled_root" >"$logs/compiler.stdout" 2>"$logs/compiler.stderr"
[ ! -s "$logs/compiler.stderr" ] || fail 'offline compiler emitted stderr'
scripts/verify-luna-bf16-kernel-set.sh "$compiled_root" \
  >"$logs/verifier.stdout" 2>"$logs/verifier.stderr"
[ ! -s "$logs/verifier.stderr" ] || fail 'compiled-set verifier emitted stderr'

stage=build-current-child
moon build --target native --release --deny-warn cmd/device_worker_child \
  >"$logs/build-child.stdout" 2>"$logs/build-child.stderr"
worker=$repo_root/_build/native/release/build/cmd/device_worker_child/device_worker_child.exe
[ -x "$worker" ] || fail 'current-source canonical device worker is missing'
worker_sha=$(sha256_file "$worker")

stage=source-stability
source_files_after=$work/source.files.after.sha256
source_links_after=$work/source.symlinks.after.v1
write_source_inventory "$source_files_after" "$source_links_after"
cmp -s "$source_files" "$source_files_after" ||
  fail 'source files changed while building the physical release'
cmp -s "$source_links" "$source_links_after" ||
  fail 'source symlinks changed while building the physical release'

stage=materialize-launch
launch_root=$work/launch
scripts/materialize-approved-tiny-bf16-launch.sh \
  "$model_root" "$compiled_root" \
  "$toolchain_manifest#sha256=$toolchain_sha" \
  "$compiler_major" "$compiler_minor" "$compiler_patch" \
  "$worker#sha256=$worker_sha" "$launch_root" \
  >"$logs/materialize.stdout" 2>"$logs/materialize.stderr"
[ ! -s "$logs/materialize.stderr" ] || fail 'launch materializer emitted stderr'
launch_sha=$(sed -n 's/^launch_sha256=//p' "$logs/materialize.stdout")
materialized_worker_sha=$(sed -n 's/^worker_sha256=//p' "$logs/materialize.stdout")
case "$launch_sha" in ''|*[!0-9a-f]*)
  fail 'materializer omitted a lowercase launch digest'
  ;;
esac
[ "${#launch_sha}" -eq 64 ] || fail 'materializer launch digest is invalid'
[ "$materialized_worker_sha" = "$worker_sha" ] ||
  fail 'materializer substituted the current-source child digest'
[ "$(sha256_file "$launch_root/bin/lunaflux-device-worker")" = "$worker_sha" ] ||
  fail 'materialized child bytes do not match the current-source child'
[ "$(sha256_file "$launch_root/lunaflux.launch.json")" = "$launch_sha" ] ||
  fail 'materialized launch bytes do not match the published digest'

stage=materialize-prefix-launch
prefix_launch_root=$work/prefix-launch
scripts/materialize-approved-tiny-bf16-launch.sh \
  "$model_root" "$compiled_root" \
  "$toolchain_manifest#sha256=$toolchain_sha" \
  "$compiler_major" "$compiler_minor" "$compiler_patch" \
  "$worker#sha256=$worker_sha" "$prefix_launch_root" prefix-reuse-v1 \
  >"$logs/materialize-prefix.stdout" 2>"$logs/materialize-prefix.stderr"
[ ! -s "$logs/materialize-prefix.stderr" ] ||
  fail 'prefix launch materializer emitted stderr'
prefix_launch_sha=$(sed -n 's/^launch_sha256=//p' "$logs/materialize-prefix.stdout")
prefix_worker_sha=$(sed -n 's/^worker_sha256=//p' "$logs/materialize-prefix.stdout")
case "$prefix_launch_sha" in ''|*[!0-9a-f]*)
  fail 'prefix materializer omitted a lowercase launch digest'
  ;;
esac
[ "${#prefix_launch_sha}" -eq 64 ] || fail 'prefix launch digest is invalid'
[ "$prefix_worker_sha" = "$worker_sha" ] ||
  fail 'prefix materializer substituted the current-source child digest'
[ "$(sha256_file "$prefix_launch_root/bin/lunaflux-device-worker")" = "$worker_sha" ] ||
  fail 'prefix launch child bytes do not match the current-source child'
[ "$(sha256_file "$prefix_launch_root/lunaflux.launch.json")" = "$prefix_launch_sha" ] ||
  fail 'prefix launch bytes do not match the published digest'
grep -Fxq 'materialization_profile=prefix-reuse-v1' "$logs/materialize-prefix.stdout" ||
  fail 'prefix launch materializer did not select the authenticated profile'

stage=materialize-host-sampling-referee
host_referee_root=$work/host-referee-launch
scripts/materialize-approved-tiny-bf16-launch.sh \
  "$model_root" "$compiled_root" \
  "$toolchain_manifest#sha256=$toolchain_sha" \
  "$compiler_major" "$compiler_minor" "$compiler_patch" \
  "$worker#sha256=$worker_sha" "$host_referee_root" host-sampling-referee-v1 \
  >"$logs/materialize-host-referee.stdout" \
  2>"$logs/materialize-host-referee.stderr"
[ ! -s "$logs/materialize-host-referee.stderr" ] ||
  fail 'host-sampling referee materializer emitted stderr'
host_referee_launch_sha=$(sed -n 's/^launch_sha256=//p' "$logs/materialize-host-referee.stdout")
[ "$(sha256_file "$host_referee_root/lunaflux.launch.json")" = "$host_referee_launch_sha" ] ||
  fail 'host-sampling referee launch digest drifted'
grep -Fxq 'materialization_profile=host-sampling-referee-v1' \
  "$logs/materialize-host-referee.stdout" ||
  fail 'host-sampling referee profile was not selected'
grep -Fq '"schema_version":"lunaflux.runtime.v3"' \
  "$host_referee_root/model-root/runtime/descriptor.json" ||
  fail 'host-sampling referee descriptor is not legacy host schema v3'
if grep -Fq 'sampling_runtime' \
  "$host_referee_root/model-root/runtime/descriptor.json"; then
  fail 'host-sampling referee descriptor retained embedded sampling'
fi

stage=build-spawned-campaign
moon build --target native --deny-warn tests/approved_model_spawned_physical \
  >"$logs/build-campaign.stdout" 2>"$logs/build-campaign.stderr"
campaign=$repo_root/_build/native/debug/build/tests/approved_model_spawned_physical/approved_model_spawned_physical.exe
[ -x "$campaign" ] || fail 'spawned physical campaign executable is missing'

stage=final-source-stability
source_files_final=$work/source.files.final.sha256
source_links_final=$work/source.symlinks.final.v1
write_source_inventory "$source_files_final" "$source_links_final"
cmp -s "$source_files" "$source_files_final" ||
  fail 'source files changed while materializing the physical campaign'
cmp -s "$source_links" "$source_links_final" ||
  fail 'source symlinks changed while materializing the physical campaign'

stage=gpu-baseline
command -v nvidia-smi >/dev/null 2>&1 || fail 'nvidia-smi is unavailable'
nvidia-smi --query-gpu=index,uuid,name,compute_cap,memory.used \
  --format=csv,noheader,nounits >"$logs/gpu-inventory.before" \
  2>"$logs/gpu-inventory.before.stderr"
[ ! -s "$logs/gpu-inventory.before.stderr" ] || fail 'GPU inventory emitted stderr'
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits >"$logs/gpu-processes.before" \
  2>"$logs/gpu-processes.before.stderr"
[ ! -s "$logs/gpu-processes.before" ] || fail 'GPU already has a compute process'
[ ! -s "$logs/gpu-processes.before.stderr" ] || fail 'GPU baseline emitted stderr'
export CUDA_VISIBLE_DEVICES=0

stage=spawned-execution
lunaflux_run_tracked_campaign "$logs/spawned.stdout" "$logs/spawned.stderr" \
  "$campaign" "$launch_root#sha256=$launch_sha" "$worker_sha" ||
  fail 'spawned execution campaign failed'
[ ! -s "$logs/spawned.stderr" ] || fail 'spawned execution emitted stderr'
for exact in \
  'schema=lunaflux-approved-model-spawned-execution.v2' \
  "worker_executable_sha256=$worker_sha" \
  'traffic_readiness=0' \
  'plan_executed=2' \
  'generated_tokens=1031,2185' \
  'cleanup_complete=1' \
  'spawned_execution=pass'; do
  grep -Fxq "$exact" "$logs/spawned.stdout" ||
    fail "spawned execution evidence lost: $exact"
done

stage=spawned-device-greedy-readback
lunaflux_run_tracked_campaign \
  "$logs/device-greedy.stdout" "$logs/device-greedy.stderr" \
  "$campaign" device-greedy \
  "$launch_root#sha256=$launch_sha" \
  "$host_referee_root#sha256=$host_referee_launch_sha" "$worker_sha" ||
  fail 'spawned embedded-device-greedy qualification failed'
[ ! -s "$logs/device-greedy.stderr" ] ||
  fail 'spawned embedded-device-greedy qualification emitted stderr'
for exact in \
  'schema=lunaflux-spawned-device-greedy-qualification.v1' \
  'spawn_boundary=descriptor_file,worker_wire,child_bootstrap,paged_device_executor' \
  'embedded_sampling_runtime=embedded_cuda_greedy_v1' \
  'host_referee_sampling_runtime=host_sampling' \
  'fused_v2_runtime=optional-absent' \
  'request_plans=2' \
  'sampling_result_row_bytes=8' \
  'sampling_result_layout=token_i32,status_i32' \
  'sampling_readback_bytes_per_plan=8' \
  'sampling_readback_total_bytes=16' \
  'sampling_success_status=-1' \
  'nonfinite_status=first_nonfinite_token_id' \
  'tie_policy=lowest_token_id' \
  'generated_tokens=1031,2185' \
  'host_oracle_tokens=1031,2185' \
  'selected_logit_correctness=pass' \
  'graph_policy_interaction=authenticated-and-stable' \
  'child_closed=2' 'cleanup_complete=1' \
  'physical_cuda_observed=true' 'qualification_only=true' \
  'manifest_bindable=false' 'promotion_authority=absent'; do
  grep -Fxq "$exact" "$logs/device-greedy.stdout" ||
    fail "spawned embedded-device-greedy evidence lost: $exact"
done
grep -Eq '^embedded_graph_path=(captured|ordered-eager)$' \
  "$logs/device-greedy.stdout" || fail 'embedded graph path was not reported'
grep -Eq '^host_graph_path=(captured|ordered-eager)$' \
  "$logs/device-greedy.stdout" || fail 'host referee graph path was not reported'
grep -Eq '^device_greedy_qualification_sha256=[0-9a-f]{64}$' \
  "$logs/device-greedy.stdout" || fail 'device-greedy canonical digest is absent'

stage=native-listener
lunaflux_run_tracked_campaign "$logs/serving.stdout" "$logs/serving.stderr" \
  "$campaign" serving "$launch_root#sha256=$launch_sha" "$worker_sha" ||
  fail 'native listener campaign failed'
[ ! -s "$logs/serving.stderr" ] || fail 'native listener execution emitted stderr'
for exact in \
  'schema=lunaflux-approved-model-serving-readiness.v1' \
  "worker_executable_sha256=$worker_sha" \
  'native_tcp_listener=pass' \
  'request_count=1' \
  'event_order=accepted,token,token,usage,completed' \
  'event_count=5' \
  'generated_tokens=1031,2185' \
  'network_accepts=1' \
  'network_disconnects=1' \
  'kv_pages_used_after_request=0' \
  'kv_pages_free_after_request=32' \
  'listener_closed=1' \
  'child_closed=1' \
  'cleanup_complete=1' \
  'serving_readiness_validation=pass'; do
  grep -Fxq "$exact" "$logs/serving.stdout" ||
    fail "native listener evidence lost: $exact"
done

stage=broad-bf16-serving-qualification
lunaflux_run_tracked_campaign \
  "$logs/broad-bf16-serving.stdout" \
  "$logs/broad-bf16-serving.stderr" \
  "$campaign" broad-serving "$launch_root#sha256=$launch_sha" "$worker_sha" ||
  fail 'broad BF16 serving qualification failed'
[ ! -s "$logs/broad-bf16-serving.stderr" ] ||
  fail 'broad BF16 serving qualification emitted stderr'
for exact in \
  'schema=lunaflux-approved-model-broad-bf16-serving-qualification.v1' \
  'campaign_scope=qualification-only' \
  'production_readiness=not-claimed' \
  'performance_baseline=not-established' \
  'benchmark_claim=not-made' \
  "worker_executable_sha256=$worker_sha" \
  'model_dtype=bf16' \
  'mixed_concurrent_requests=pass' \
  'concurrent_request_count=2' \
  'saturation_backpressure=pass' \
  'overload_rejection=not-exercised-single-connection-owner' \
  'cancellation_isolation=pass' \
  'cancellation_count=1' \
  'typed_foreign_model_rejection=pass' \
  'malformed_native_frame=fail-closed-connection-isolation-pass' \
  'malformed_http=not-exercised-native-framed-owner' \
  'post_malformed_same_owner_recovery=pass' \
  'pre_malformed_reconnect_after_rejection=pass' \
  'fresh_owner_restart=pass' \
  'drain_transition=pass' \
  'request_count=5' \
  'completion_count=4' \
  'event_count=22' \
  'network_accepts=5' \
  'network_disconnects=5' \
  'network_rejections=2' \
  'kv_pages_used_after_cases=0' \
  'kv_pages_free_after_cases=32' \
  'restart_request_event_count=5' \
  'restart_network_accepts=1' \
  'restart_network_disconnects=1' \
  'restart_kv_pages_free=32' \
  'listener_closed=2' \
  'child_closed=2' \
  'cleanup_complete=1' \
  'broad_bf16_serving_qualification=pass'; do
  grep -Fxq "$exact" "$logs/broad-bf16-serving.stdout" ||
    fail "broad BF16 serving evidence lost: $exact"
done
grep -Eq '^backpressure_count=[1-9][0-9]*$' \
  "$logs/broad-bf16-serving.stdout" ||
  fail 'broad BF16 serving qualification omitted physical backpressure'
grep -Eq '^broad_bf16_serving_qualification_sha256=[0-9a-f]{64}$' \
  "$logs/broad-bf16-serving.stdout" ||
  fail 'broad BF16 serving qualification omitted its canonical digest'

stage=openai-responses-qualification
qualification_credential='lunaflux-physical-qualification-v1'
lunaflux_run_tracked_campaign \
  "$logs/openai-qualification.stdout" \
  "$logs/openai-qualification.stderr" \
  "$campaign" openai-qualification \
  "$launch_root#sha256=$launch_sha" "$worker_sha" \
  "$qualification_credential" ||
  fail 'OpenAI Responses loopback qualification failed'
[ ! -s "$logs/openai-qualification.stderr" ] ||
  fail 'OpenAI Responses loopback qualification emitted stderr'
for exact in \
  'schema=lunaflux-openai-responses-loopback-qualification.v1' \
  "worker_executable_sha256=$worker_sha" \
  'qualification_provenance=non_routable' \
  'production_readiness=not_ready' \
  'transport=loopback_plaintext' \
  'bearer_auth_missing=401' \
  'bearer_auth_wrong=401' \
  'responses_status=200' \
  'event_order=response.created,response.output_text.delta,response.usage,response.completed,done' \
  'event_count=5' \
  'request_count=1' \
  'completion_count=1' \
  'network_accepts=3' \
  'network_disconnects=3' \
  'network_rejections=2' \
  'kv_pages_used_after_request=0' \
  'kv_pages_free_after_request=32' \
  'drain_transition=pass' \
  'post_drain_admission=refused' \
  'listener_closed=1' \
  'child_closed=1' \
  'cleanup_complete=1' \
  'health_endpoint=loopback-200-exercised' \
  'readiness_endpoint=loopback-503-qualification' \
  'tls_validation=not-run' \
  'openai_responses_qualification=pass'; do
  grep -Fxq "$exact" "$logs/openai-qualification.stdout" ||
    fail "OpenAI Responses qualification evidence lost: $exact"
done
grep -Eq '^openai_responses_qualification_sha256=[0-9a-f]{64}$' \
  "$logs/openai-qualification.stdout" ||
  fail 'OpenAI Responses qualification omitted its canonical digest'

stage=physical-benchmark
lunaflux_run_tracked_campaign "$logs/benchmark.stdout" "$logs/benchmark.stderr" \
  "$campaign" benchmark "$launch_root#sha256=$launch_sha" "$worker_sha" ||
  fail 'physical benchmark campaign failed'
[ ! -s "$logs/benchmark.stderr" ] || fail 'physical benchmark emitted stderr'
for exact in \
  'schema=lunaflux-physical-benchmark-trial.v1' \
  'engine=lunaflux' \
  'profile=latency' \
  'workload_contract=pinned-1-input-2-output-qualification' \
  'submitted=1' \
  'completed=1' \
  'rejected=0' \
  'timed_out=0' \
  'cancelled=0' \
  'failed=0' \
  'input_tokens=1' \
  'generated_tokens=2' \
  'event_order=accepted,token,token,usage,completed' \
  'comparison_admission=not-run' \
  'cleanup_complete=1'; do
  grep -Fxq "$exact" "$logs/benchmark.stdout" ||
    fail "physical benchmark evidence lost: $exact"
done
grep -Eq '^physical_benchmark_sha256=[0-9a-f]{64}$' "$logs/benchmark.stdout" ||
  fail 'physical benchmark omitted its canonical digest'

stage=physical-prefix-reuse
lunaflux_run_tracked_campaign "$logs/prefix.stdout" "$logs/prefix.stderr" \
  "$campaign" prefix-reuse \
  "$prefix_launch_root#sha256=$prefix_launch_sha" "$worker_sha" ||
  fail 'physical prefix-reuse campaign failed'
[ ! -s "$logs/prefix.stderr" ] || fail 'physical prefix reuse emitted stderr'
for exact in \
  'schema=lunaflux-approved-model-prefix-reuse.v1' \
  'independent_expected_tokens=1355,1240' \
  'observed_tokens=1355,1240' \
  'first_cached_input_tokens=0' \
  'second_cached_input_tokens=8' \
  'prefix_lookups=2' \
  'prefix_hits=1' \
  'prefix_misses=1' \
  'prefix_publications=1' \
  'prefix_tokens_reused=8' \
  'retained_prefix_pages_before_close=1' \
  'owner_closed=1' \
  'child_closed=1' \
  'cleanup_complete=1' \
  'prefix_reuse_qualification=pass'; do
  grep -Fxq "$exact" "$logs/prefix.stdout" ||
    fail "physical prefix-reuse evidence lost: $exact"
done
grep -Eq '^prefix_reuse_sha256=[0-9a-f]{64}$' "$logs/prefix.stdout" ||
  fail 'physical prefix reuse omitted its canonical digest'

stage=physical-context-churn
lunaflux_run_tracked_campaign \
  "$logs/context-churn.stdout" "$logs/context-churn.stderr" \
  "$campaign" context-churn \
  "$launch_root#sha256=$launch_sha" "$worker_sha" ||
  fail 'physical context-churn qualification failed'
[ ! -s "$logs/context-churn.stderr" ] ||
  fail 'physical context-churn qualification emitted stderr'
for exact in \
  'schema=lunaflux-approved-model-bf16-context-churn-qualification.v1' \
  'campaign_scope=qualification-only' \
  'release_promotion=not-claimed' \
  'broader_context_length_coverage=not-established' \
  'selected_logit_correctness=not-observed' \
  'gpu_allocator_instrumentation=not-observed-by-this-mode' \
  'upstream_revision=3579d71fd57e04f5a364d824d3a2ec3e913dbb67' \
  'runtime_recipe=dense_llama_paged_aot_v5' \
  'model_content_sha256=852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30' \
  'tokenizer_sha256=fbcdbe15960e43ef351662e7b77a319ceb294b3c5dc2569c23b729fb87e13d7b' \
  "worker_executable_sha256=$worker_sha" \
  'model_dtype=bf16' \
  'input_tokens_per_request=1' \
  'output_tokens_per_request=2' \
  'actual_context_tokens_per_request=3' \
  'independent_expected_tokens=1031,2185' \
  'declared_context_ceiling_matrix=8,32,256' \
  'fresh_child_cycles=3' \
  'requests_per_child=3' \
  'request_count=9' \
  'event_count=45' \
  'plan_count=18' \
  'network_observation=structural-no-listener-owner' \
  'listener_owner_count=0' \
  'network_accepts=0' \
  'network_disconnects=0' \
  'network_rejections=0' \
  'kv_pages_used_after_each_request=0' \
  'kv_pages_free_after_each_request=32' \
  'children_spawned=3' \
  'children_closed=3' \
  'cleanup_complete=1' \
  'context_churn_qualification=pass'; do
  grep -Fxq "$exact" "$logs/context-churn.stdout" ||
    fail "physical context-churn evidence lost: $exact"
done
grep -Eq '^model_plan_sha256=[0-9a-f]{64}$' \
  "$logs/context-churn.stdout" ||
  fail 'physical context-churn evidence omitted its model-plan digest'
grep -Eq '^release_preflight_sha256=[0-9a-f]{64}$' \
  "$logs/context-churn.stdout" ||
  fail 'physical context-churn evidence omitted its release digest'
verify_canonical_evidence \
  "$logs/context-churn.stdout" context_churn_qualification_sha256 36 \
  'physical context-churn qualification'

stage=physical-long-context
lunaflux_run_tracked_campaign \
  "$logs/long-context.stdout" "$logs/long-context.stderr" \
  "$campaign" long-context \
  "$launch_root#sha256=$launch_sha" "$worker_sha" ||
  fail 'physical actual long-context qualification failed'
[ ! -s "$logs/long-context.stderr" ] ||
  fail 'physical actual long-context qualification emitted stderr'
for exact in \
  'schema=lunaflux-approved-model-bf16-actual-long-context-qualification.v1' \
  'campaign_scope=qualification-only' \
  'release_promotion=not-claimed' \
  'required_launch_envelope=max_input_tokens_9' \
  'actual_input_token_lengths=8,9' \
  'actual_context_tokens_including_output=9,10' \
  'physical_page_span=1,2' \
  'beyond_9_token_input=not-established' \
  '128_or_255_token_execution=not-admitted' \
  'full_256_token_context=not-established' \
  'selected_logit_correctness=not-observed' \
  'gpu_allocator_instrumentation=not-observed-by-this-mode' \
  'upstream_revision=3579d71fd57e04f5a364d824d3a2ec3e913dbb67' \
  'runtime_recipe=dense_llama_paged_aot_v5' \
  'model_content_sha256=852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30' \
  'tokenizer_sha256=fbcdbe15960e43ef351662e7b77a319ceb294b3c5dc2569c23b729fb87e13d7b' \
  "worker_executable_sha256=$worker_sha" \
  'referee_schema=lunaflux.approved-bf16-prefix-referee.v1' \
  'referee_sha256=765d4b80750e377fea131ce140a0a67931724ab06c24dc5e322c7e6b295ab8e3' \
  'referee_path=approved_scalar_bf16_reference_executor' \
  'independent_expected_tokens=1355,1240' \
  'observed_tokens_per_cycle=1355,1240' \
  'cache_permission=disabled' \
  'prefix_reuse=not-exercised' \
  'fresh_child_cycles=3' \
  'request_count=6' \
  'requests_per_cycle=2' \
  'events_per_cycle=8' \
  'prefill_plans_per_cycle=17' \
  'plan_sequence_range_per_cycle=1-17' \
  'event_count=24' \
  'plan_count=51' \
  'input_tokens_physically_processed=51' \
  'network_observation=structural-no-listener-owner' \
  'listener_owner_count=0' \
  'network_accepts=0' \
  'network_disconnects=0' \
  'network_rejections=0' \
  'kv_pages_used_after_each_request=0' \
  'kv_pages_free_after_each_request=32' \
  'children_spawned=3' \
  'children_closed=3' \
  'cleanup_complete=1' \
  'actual_long_context_qualification=pass'; do
  grep -Fxq "$exact" "$logs/long-context.stdout" ||
    fail "physical actual long-context evidence lost: $exact"
done
grep -Eq '^model_plan_sha256=[0-9a-f]{64}$' \
  "$logs/long-context.stdout" ||
  fail 'physical actual long-context evidence omitted its model-plan digest'
grep -Eq '^release_preflight_sha256=[0-9a-f]{64}$' \
  "$logs/long-context.stdout" ||
  fail 'physical actual long-context evidence omitted its release digest'
verify_canonical_evidence \
  "$logs/long-context.stdout" actual_long_context_qualification_sha256 47 \
  'physical actual long-context qualification'

stage=final-balance
nvidia-smi --query-gpu=index,uuid,name,compute_cap,memory.used \
  --format=csv,noheader,nounits >"$logs/gpu-inventory.after" \
  2>"$logs/gpu-inventory.after.stderr"
[ ! -s "$logs/gpu-inventory.after.stderr" ] || fail 'final GPU inventory emitted stderr'
cmp -s "$logs/gpu-inventory.before" "$logs/gpu-inventory.after" ||
  fail 'GPU inventory or framebuffer use changed across campaign'
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits >"$logs/gpu-processes.after" \
  2>"$logs/gpu-processes.after.stderr"
[ ! -s "$logs/gpu-processes.after" ] || fail 'GPU compute process survived campaign'
[ ! -s "$logs/gpu-processes.after.stderr" ] || fail 'final GPU query emitted stderr'
ps_snapshot=$work/processes.after
ps -eo pid=,args= >"$ps_snapshot" || fail 'final process inventory failed'
if grep -F "$launch_root/bin/lunaflux-device-worker" "$ps_snapshot" \
  >"$logs/child-processes.after"; then
  fail 'spawned child process survived campaign'
else
  grep_status=$?
  [ "$grep_status" -eq 1 ] || fail 'spawned child process search failed'
fi
if grep -F "$prefix_launch_root/bin/lunaflux-device-worker" "$ps_snapshot" \
  >"$logs/prefix-child-processes.after"; then
  fail 'prefix child process survived campaign'
else
  grep_status=$?
  [ "$grep_status" -eq 1 ] || fail 'prefix child process search failed'
fi
if grep -F "$host_referee_root/bin/lunaflux-device-worker" "$ps_snapshot" \
  >"$logs/host-referee-child-processes.after"; then
  fail 'host referee child process survived campaign'
else
  grep_status=$?
  [ "$grep_status" -eq 1 ] || fail 'host referee child process search failed'
fi

stage=complete
campaign_complete=1
