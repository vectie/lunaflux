#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
TZ=UTC
export LC_ALL TZ
umask 077

fail() { printf 'spawned device-greedy campaign rejected: %s\n' "$1" >&2; exit 1; }
if [[ $# != 7 ]]; then
  printf 'usage: %s ABSOLUTE_COMPUTE_SANITIZER ABSOLUTE_NVIDIA_SMI EMBEDDED_LAUNCH#sha HOST_LAUNCH#sha WORKER_SHA ABSOLUTE_NEW_OUTPUT EXPECTED_FUSED_MODE\n' "$0" >&2
  exit 2
fi
sanitizer=$1
nvidia_smi=$2
embedded_argument=$3
host_argument=$4
worker_sha=$5
output=$6
fused_mode=$7
[[ $fused_mode == optional-absent || $fused_mode == required-present ]] ||
  fail 'fused mode is not exact'
for tool in "$sanitizer" "$nvidia_smi"; do
  [[ $tool == /* && -f $tool && ! -L $tool && -x $tool && \
    $(realpath -- "$tool") == "$tool" ]] || fail 'tool path is not canonical'
done
for argument in "$embedded_argument" "$host_argument"; do
  [[ $argument == /*#sha256=* ]] || fail 'launch is not independently pinned'
done
embedded=${embedded_argument%#sha256=*}
embedded_sha=${embedded_argument##*#sha256=}
host=${host_argument%#sha256=*}
host_sha=${host_argument##*#sha256=}
[[ $worker_sha =~ ^[0-9a-f]{64}$ && $embedded_sha =~ ^[0-9a-f]{64}$ && \
  $host_sha =~ ^[0-9a-f]{64}$ ]] || fail 'input digest is malformed'
for root in "$embedded" "$host"; do
  [[ -d $root && ! -L $root && $(realpath -- "$root") == "$root" ]] ||
    fail 'launch root is not canonical'
done
[[ $output == /* && ! -e $output && ! -L $output ]] || fail 'output is not new'
parent=${output%/*}
name=${output##*/}
[[ $name =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ && -d $parent && \
  ! -L $parent && $(realpath -- "$parent") == "$parent" ]] ||
  fail 'output path is not canonical'

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
[[ $(sha256_file "$embedded/lunaflux.launch.json") == "$embedded_sha" && \
  $(sha256_file "$host/lunaflux.launch.json") == "$host_sha" ]] ||
  fail 'launch digest pin mismatch'
[[ $(sha256_file "$embedded/bin/lunaflux-device-worker") == "$worker_sha" && \
  $(sha256_file "$host/bin/lunaflux-device-worker") == "$worker_sha" ]] ||
  fail 'spawned child executable pin mismatch'

stage=$(mktemp -d "$parent/.${name}.partial.XXXXXX")
stage=$(CDPATH= cd -- "$stage" && pwd -P)
cleanup() {
  if [[ -n ${stage:-} && -d $stage ]]; then
    chmod -R u+rwX "$stage" 2>/dev/null || true
    rm -rf -- "$stage"
  fi
}
trap cleanup EXIT HUP INT TERM
logs=$stage/logs
mkdir "$logs"

moon build --target native --deny-warn tests/approved_model_spawned_physical \
  >"$logs/build.stdout" 2>"$logs/build.stderr" || fail 'campaign build failed'
[[ ! -s $logs/build.stderr ]] || fail 'campaign build emitted stderr'
campaign=$root/_build/native/debug/build/tests/approved_model_spawned_physical/approved_model_spawned_physical.exe
[[ -x $campaign && ! -L $campaign ]] || fail 'campaign executable is absent'
campaign_sha=$(sha256_file "$campaign")
sanitizer_sha=$(sha256_file "$sanitizer")
nvidia_smi_sha=$(sha256_file "$nvidia_smi")

"$nvidia_smi" --query-gpu=index,uuid,name,compute_cap,memory.used \
  --format=csv,noheader,nounits >"$logs/gpu.before" 2>"$logs/gpu.before.stderr"
[[ ! -s $logs/gpu.before.stderr ]] || fail 'GPU baseline emitted stderr'
"$nvidia_smi" --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits >"$logs/processes.before" 2>"$logs/processes.before.stderr"
[[ ! -s $logs/processes.before && ! -s $logs/processes.before.stderr ]] ||
  fail 'GPU was not idle before qualification'
export CUDA_VISIBLE_DEVICES=0
mode=device-greedy
[[ $fused_mode == required-present ]] && mode=device-greedy-fused-v2
. "$root/scripts/physical-campaign-process-group.sh"
for check in memcheck racecheck initcheck; do
  leak=()
  [[ $check == memcheck ]] && leak=(--leak-check full)
  lunaflux_run_tracked_campaign "$logs/$check.stdout" "$logs/$check.stderr" \
    "$sanitizer" --tool "$check" "${leak[@]}" --target-processes all \
    --error-exitcode 99 --log-file "$logs/$check.%p.log" \
    "$campaign" "$mode" "$embedded_argument" "$host_argument" "$worker_sha" ||
    fail "$check execution failed"
  [[ ! -s $logs/$check.stderr ]] || fail "$check execution emitted stderr"
  report_count=0
  while IFS= read -r report; do
    report_count=$((report_count + 1))
    [[ $(grep -Fxc '========= ERROR SUMMARY: 0 errors' "$report") == 1 ]] ||
      fail "$check reported an error"
  done < <(find "$logs" -maxdepth 1 -name "$check.*.log" -type f | sort)
  [[ $report_count -ge 2 ]] || fail "$check did not observe both spawned CUDA children"
done
cmp -s "$logs/memcheck.stdout" "$logs/racecheck.stdout" &&
  cmp -s "$logs/memcheck.stdout" "$logs/initcheck.stdout" ||
  fail 'sanitizer executions produced different observations'
runtime=$logs/memcheck.stdout
for exact in \
  'schema=lunaflux-spawned-device-greedy-qualification.v1' \
  'spawn_boundary=descriptor_file,worker_wire,child_bootstrap,paged_device_executor' \
  'sampling_result_row_bytes=8' 'sampling_result_layout=token_i32,status_i32' \
  'sampling_readback_total_bytes=16' 'sampling_success_status=-1' \
  'nonfinite_status=first_nonfinite_token_id' 'tie_policy=lowest_token_id' \
  'generated_tokens=1031,2185' 'host_oracle_tokens=1031,2185' \
  'selected_logit_correctness=pass' "fused_v2_runtime=$fused_mode" \
  'child_closed=2' 'cleanup_complete=1' 'physical_cuda_observed=true' \
  'qualification_only=true' 'manifest_bindable=false' 'promotion_authority=absent'; do
  grep -Fx "$exact" "$runtime" >/dev/null || fail "runtime evidence drifted: $exact"
done
grep -Eq '^embedded_graph_path=(captured|ordered-eager)$' "$runtime" ||
  fail 'embedded graph policy path is absent'

"$nvidia_smi" --query-gpu=index,uuid,name,compute_cap,memory.used \
  --format=csv,noheader,nounits >"$logs/gpu.after" 2>"$logs/gpu.after.stderr"
cmp -s "$logs/gpu.before" "$logs/gpu.after" || fail 'GPU resource balance drifted'
"$nvidia_smi" --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits >"$logs/processes.after" 2>"$logs/processes.after.stderr"
[[ ! -s $logs/processes.after && ! -s $logs/processes.after.stderr && \
  ! -s $logs/gpu.after.stderr ]] || fail 'GPU process survived qualification'

. "$root/scripts/immutable-evidence-directory.sh"
lunaflux_prepare_evidence_manifest "$stage" || fail 'FILES manifest failed'
files_sha=$lunaflux_evidence_manifest_sha256
printf '%s\n' 'schema=lunaflux-spawned-device-greedy-physical-campaign.v1' \
  'outcome=spawned-device-greedy-physical-pass' \
  "campaign_executable_sha256=$campaign_sha" "worker_executable_sha256=$worker_sha" \
  "embedded_launch_sha256=$embedded_sha" "host_launch_sha256=$host_sha" \
  "compute_sanitizer_sha256=$sanitizer_sha" "nvidia_smi_sha256=$nvidia_smi_sha" \
  "fused_v2_runtime=$fused_mode" 'request_plans=2' 'readback_bytes=16' \
  'host_referee=full-logits-production-route' 'memcheck_errors=0' \
  'racecheck_errors=0' 'initcheck_errors=0' 'resources=closed' \
  'adversarial_tie_policy=source-and-host-contract-only' \
  'adversarial_nonfinite_policy=source-and-host-contract-only' \
  'physical_cuda_observed=true' 'qualification_only=true' \
  'promotion_authority=absent' "evidence_files_manifest_sha256=$files_sha" \
  >"$stage/RESULT.txt"
{
  printf '%s  FILES.sha256\n' "$(sha256_file "$stage/FILES.sha256")"
  printf '%s  RESULT.txt\n' "$(sha256_file "$stage/RESULT.txt")"
} >"$stage/OUTER_SEAL.sha256"
outer_sha=$(sha256_file "$stage/OUTER_SEAL.sha256")
lunaflux_seal_evidence_directory "$stage" || fail 'evidence seal failed'
[[ ! -e $output && ! -L $output ]] || fail 'output appeared before publication'
chmod 0755 "$stage"
mv -- "$stage" "$output"
chmod 0555 "$output"
stage=
trap - EXIT HUP INT TERM
"$root/scripts/verify-spawned-device-greedy-physical-campaign.sh" \
  "$output" "$outer_sha" >/dev/null || fail 'published evidence rejected'
printf 'outcome=spawned-device-greedy-physical-campaign-published outer_seal_sha256=%s authority=qualification-only\n' "$outer_sha"
