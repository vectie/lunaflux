#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
TZ=UTC
export LC_ALL TZ
umask 077

fail() {
  printf 'Qwen3 v12 serving campaign rejected: %s\n' "$1" >&2
  exit 1
}

usage() {
  printf '%s\n' \
    "usage: $0 ABSOLUTE_LUNAFLUX#sha256=HEX ABSOLUTE_DEPLOYMENT_ROOT#sha256=HEX ABSOLUTE_REFERENCE_CORPUS#sha256=HEX ABSOLUTE_NVIDIA_SMI#sha256=HEX ABSOLUTE_CURL#sha256=HEX OPENAI_SSE_PHASE BENCHMARK_CAPACITY ABSOLUTE_NEW_OUTPUT" >&2
  exit 2
}

[[ $# -eq 8 ]] || usage
lunaflux_argument=$1
deployment_argument=$2
corpus_argument=$3
nvidia_smi_argument=$4
curl_argument=$5
openai_sse_phase=$6
benchmark_capacity=$7
output=$8

[[ $openai_sse_phase == unavailable ]] ||
  fail 'authenticated Qwen token-ID SSE benchmark bridge is unavailable'
[[ $benchmark_capacity == c1-correctness-only ]] ||
  fail 'native correctness campaign requires the separately bound c1 release'

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"
. "$repo_root/scripts/immutable-evidence-directory.sh"

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

is_sha256() { [[ $1 =~ ^[0-9a-f]{64}$ ]]; }

split_pinned() {
  local argument=$1
  [[ $argument == /*#sha256=* ]] || usage
  pinned_path=${argument%#sha256=*}
  pinned_sha=${argument##*#sha256=}
  is_sha256 "$pinned_sha" || fail 'a pinned digest is malformed'
}

require_pinned_file() {
  local argument=$1 label=$2
  split_pinned "$argument"
  [[ -f $pinned_path && ! -L $pinned_path &&
     $(realpath -- "$pinned_path") == "$pinned_path" ]] ||
    fail "$label is not a canonical regular file"
  [[ $(sha256_file "$pinned_path") == "$pinned_sha" ]] ||
    fail "$label digest differs"
}

require_new_output() {
  local target=$1 parent name
  [[ $target == /* && ! -e $target && ! -L $target ]] ||
    fail 'evidence output is not a new absolute path'
  parent=${target%/*}
  name=${target##*/}
  [[ $name =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ &&
     -d $parent && ! -L $parent && $(realpath -- "$parent") == "$parent" &&
     $target == "$parent/$name" ]] || fail 'evidence output is not canonical'
}

require_pinned_file "$lunaflux_argument" 'LunaFlux executable'
lunaflux=$pinned_path
lunaflux_sha=$pinned_sha
[[ -x $lunaflux ]] || fail 'LunaFlux executable is not executable'

split_pinned "$deployment_argument"
deployment=$pinned_path
launch_sha=$pinned_sha
[[ -d $deployment && ! -L $deployment &&
   $(realpath -- "$deployment") == "$deployment" ]] ||
  fail 'deployment root is not canonical'
launch=$deployment/lunaflux.launch.json
[[ -f $launch && ! -L $launch && $(sha256_file "$launch") == "$launch_sha" ]] ||
  fail 'deployment launch digest differs'

require_pinned_file "$corpus_argument" 'reference corpus'
corpus=$pinned_path
corpus_sha=$pinned_sha
[[ $corpus == "$deployment/corpus/reference.json" ]] ||
  fail 'reference corpus is not the deployment-published corpus'

require_pinned_file "$nvidia_smi_argument" 'nvidia-smi'
nvidia_smi=$pinned_path
nvidia_smi_sha=$pinned_sha
[[ -x $nvidia_smi ]] || fail 'nvidia-smi is not executable'
require_pinned_file "$curl_argument" 'curl'
curl=$pinned_path
curl_sha=$pinned_sha
[[ -x $curl ]] || fail 'curl is not executable'
require_new_output "$output"
command -v setsid >/dev/null 2>&1 || fail 'setsid is unavailable'

output_parent=${output%/*}
output_name=${output##*/}
stage=$(mktemp -d "$output_parent/.${output_name}.partial.XXXXXX")
stage=$(realpath -- "$stage")
logs=$stage/logs
mkdir "$logs"
published=0
supervisor_pid=
supervisor_pgid=

stop_group() {
  if [[ -n ${supervisor_pgid:-} ]] &&
    kill -0 -- "-$supervisor_pgid" 2>/dev/null; then
    kill -TERM -- "-$supervisor_pgid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 -- "-$supervisor_pgid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-$supervisor_pgid" 2>/dev/null; then
      kill -KILL -- "-$supervisor_pgid" 2>/dev/null || true
    fi
  fi
  if [[ -n ${supervisor_pid:-} ]]; then
    wait "$supervisor_pid" 2>/dev/null || true
  fi
  supervisor_pid=
  supervisor_pgid=
}

cleanup() {
  stop_group || true
  if [[ $published -ne 1 && -d ${stage:-} ]]; then
    chmod -R u+rwX "$stage" 2>/dev/null || true
    rm -rf -- "$stage"
  fi
}
trap cleanup EXIT HUP INT TERM

moon build tests/qwen3_bf16_physical tests/qwen3_bf16_serving_supervisor \
  --target native --release --deny-warn --warn-list +73 \
  >"$logs/build.stdout" 2>"$logs/build.stderr" || fail 'campaign build failed'
[[ ! -s $logs/build.stderr ]] || fail 'campaign build emitted stderr'
client=$repo_root/_build/native/release/build/tests/qwen3_bf16_physical/qwen3_bf16_physical.exe
supervisor=$repo_root/_build/native/release/build/tests/qwen3_bf16_serving_supervisor/qwen3_bf16_serving_supervisor.exe
for executable in "$client" "$supervisor"; do
  [[ -f $executable && ! -L $executable && -x $executable ]] ||
    fail 'a campaign executable is absent'
done
client_sha=$(sha256_file "$client")
supervisor_sha=$(sha256_file "$supervisor")

"$lunaflux" validate-release "$deployment_argument" \
  >"$logs/preflight.stdout" 2>"$logs/preflight.stderr" ||
  fail 'published Qwen release preflight failed'
[[ ! -s $logs/preflight.stderr ]] || fail 'release preflight emitted stderr'
for exact in \
  'schema=lunaflux-release-preflight.v1' \
  'runtime_recipe=dense_qwen3_bf16_paged_aot_v12' \
  "launch_sha256=$launch_sha" \
  'device_opened=0' 'compiler_jit_authority=0'; do
  grep -Fxq "$exact" "$logs/preflight.stdout" ||
    fail "release preflight lost: $exact"
done

preflight_value() {
  local name=$1 value count
  value=$(sed -n "s/^$name=//p" "$logs/preflight.stdout")
  count=$(grep -c "^$name=" "$logs/preflight.stdout")
  [[ $count -eq 1 && -n $value ]] || fail "preflight field differs: $name"
  printf '%s\n' "$value"
}
model_content_sha=$(preflight_value model_content_sha256)
model_plan_sha=$(preflight_value model_plan_sha256)
worker_sha=$(preflight_value worker_executable_sha256)
for digest in "$model_content_sha" "$model_plan_sha" "$worker_sha"; do
  is_sha256 "$digest" || fail 'preflight emitted a malformed digest'
done
worker=$deployment/bin/lunaflux-device-worker
[[ -x $worker && -f $worker && ! -L $worker &&
   $(sha256_file "$worker") == "$worker_sha" ]] ||
  fail 'published worker executable differs'

descriptor=$deployment/model-root/runtime/descriptor.json
policy=$deployment/policy-root/instance/policy.json
for file in "$descriptor" "$policy"; do
  [[ -f $file && ! -L $file ]] || fail 'published runtime file is absent'
done
config_sha=$(sed -n 's/.*"config_sha256":"\([0-9a-f]\{64\}\)".*/\1/p' "$descriptor")
total_pages=$(sed -n 's/.*"total_page_count":\([0-9][0-9]*\).*/\1/p' "$descriptor")
max_token_id=$(sed -n 's/.*"max_token_id":\([0-9][0-9]*\).*/\1/p' "$descriptor")
max_context=$(sed -n 's/.*"max_sequence_tokens":\([0-9][0-9]*\).*/\1/p' "$descriptor")
is_sha256 "$config_sha" || fail 'descriptor config digest is malformed'
[[ $total_pages =~ ^[1-9][0-9]*$ && $max_token_id =~ ^[1-9][0-9]*$ &&
   $max_context =~ ^[1-9][0-9]*$ ]] || fail 'descriptor serving geometry is malformed'
grep -Fq '"family":"qwen3"' "$descriptor" || fail 'descriptor family is not Qwen3'
grep -Fq '"max_batch_rows":1' "$descriptor" ||
  fail 'correctness descriptor is not c1'
grep -Fq '"max_active_requests":1' "$policy" ||
  fail 'correctness policy is not c1'

"$client" plan-check "$deployment/model-root" "$config_sha" \
  "$model_content_sha" "$model_plan_sha" \
  >"$logs/plan-check.stdout" 2>"$logs/plan-check.stderr" ||
  fail 'Qwen 28-layer plan check failed'
[[ ! -s $logs/plan-check.stderr ]] || fail 'Qwen plan check emitted stderr'
grep -Fq 'outcome=qwen3-serving-plan-pass model_family=qwen3 layer_count=28 ' \
  "$logs/plan-check.stdout" || fail 'Qwen 28-layer plan evidence is absent'

"$nvidia_smi" --query-gpu=index,uuid,name,compute_cap,memory.used \
  --format=csv,noheader,nounits >"$logs/gpu.before" 2>"$logs/gpu.before.stderr"
[[ ! -s $logs/gpu.before.stderr ]] || fail 'GPU baseline emitted stderr'
"$nvidia_smi" --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits >"$logs/processes.before" \
  2>"$logs/processes.before.stderr"
[[ ! -s $logs/processes.before && ! -s $logs/processes.before.stderr ]] ||
  fail 'GPU is not idle; exactly one engine server cannot be established'
export CUDA_VISIBLE_DEVICES=0

runtime_stdout=$logs/runtime.stdout
runtime_stderr=$logs/runtime.stderr
drain_trigger=$stage/drain.request
setsid "$supervisor" "$lunaflux" "$deployment_argument" \
  "$runtime_stdout" "$runtime_stderr" "$drain_trigger" \
  >"$logs/supervisor.stdout" 2>"$logs/supervisor.stderr" &
supervisor_pid=$!
supervisor_pgid=$supervisor_pid

ready=0
for _ in $(seq 1 6000); do
  if [[ -f $runtime_stdout ]] &&
    grep -Fxq 'readiness: true' "$runtime_stdout" &&
    grep -Fxq 'runtime_protocol=native-framed-v1' "$runtime_stdout" &&
    grep -q '^runtime_origin=' "$runtime_stdout" &&
    grep -q '^control_origin=' "$runtime_stdout"; then
    ready=1
    break
  fi
  kill -0 "$supervisor_pid" 2>/dev/null || break
  sleep 0.1
done
[[ $ready -eq 1 ]] || fail 'published Qwen runtime did not become ready'
for name in runtime_origin control_origin; do
  [[ $(grep -c "^$name=" "$runtime_stdout") -eq 1 ]] ||
    fail "$name publication is absent or duplicated"
done
runtime_origin=$(sed -n 's/^runtime_origin=//p' "$runtime_stdout")
control_origin=$(sed -n 's/^control_origin=//p' "$runtime_stdout")
[[ $runtime_origin == luna+tcp://127.0.0.1:* ]] ||
  fail 'native runtime origin is not exact loopback'
[[ $control_origin == http://127.0.0.1:* && $control_origin != "$runtime_origin" ]] ||
  fail 'native control origin is not the separate loopback control listener'
runtime_address=${runtime_origin#luna+tcp://}

"$curl" --fail --silent --show-error --max-time 30 \
  "$control_origin/healthz" -o "$logs/health.body" ||
  fail 'Qwen health endpoint failed'
"$curl" --fail --silent --show-error --max-time 30 \
  "$control_origin/readyz" -o "$logs/readiness.body" ||
  fail 'Qwen readiness endpoint failed'

"$client" serve-check "$runtime_address" "$corpus" "$corpus_sha" \
  "$model_content_sha" "$model_plan_sha" "$max_token_id" "$max_context" \
  >"$logs/serve-check.stdout" 2>"$logs/serve-check.stderr" ||
  fail 'persistent Qwen greedy corpus request failed'
[[ ! -s $logs/serve-check.stderr ]] || fail 'Qwen corpus request emitted stderr'
grep -Fq 'outcome=qwen3-serving-corpus-pass ' "$logs/serve-check.stdout" ||
  fail 'Qwen greedy corpus output did not match'
input_tokens=$(sed -n 's/.* input_tokens=\([0-9][0-9]*\) .*/\1/p' "$logs/serve-check.stdout")
output_tokens=$(sed -n 's/.* output_tokens=\([0-9][0-9]*\)$/\1/p' "$logs/serve-check.stdout")
[[ $input_tokens =~ ^[1-9][0-9]*$ && $output_tokens =~ ^[1-9][0-9]*$ ]] ||
  fail 'Qwen corpus token counts are malformed'

"$nvidia_smi" --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits >"$logs/processes.during" \
  2>"$logs/processes.during.stderr"
[[ ! -s $logs/processes.during.stderr &&
   $(wc -l <"$logs/processes.during" | tr -d ' ') -eq 1 ]] ||
  fail 'exactly one Qwen engine GPU process was not resident'

metrics_balanced=0
for _ in $(seq 1 300); do
  if "$curl" --fail --silent --show-error --max-time 30 \
    "$control_origin/metrics" -o "$logs/metrics.json"; then
    metrics_sha=$(sha256_file "$logs/metrics.json")
    if "$client" metrics-check "$logs/metrics.json" "$metrics_sha" \
      "$input_tokens" "$output_tokens" "$total_pages" \
      >"$logs/metrics-check.stdout" 2>"$logs/metrics-check.stderr"; then
      metrics_balanced=1
      break
    fi
  fi
  sleep 0.1
done
[[ $metrics_balanced -eq 1 ]] || fail 'Qwen serving resource metrics differ'
[[ ! -s $logs/metrics-check.stderr ]] || fail 'metrics check emitted stderr'
grep -Fq 'outcome=qwen3-serving-metrics-pass ' "$logs/metrics-check.stdout" ||
  fail 'terminal request resource balance is absent'

: >"$drain_trigger"
supervisor_status=0
wait "$supervisor_pid" || supervisor_status=$?
supervisor_pid=
supervisor_pgid=
[[ $supervisor_status -eq 0 ]] || fail 'Qwen runtime supervisor failed'
[[ ! -s $logs/supervisor.stderr && ! -s $runtime_stderr ]] ||
  fail 'Qwen owner or supervisor emitted stderr'
for exact in \
  'schema=lunaflux-qwen3-native-supervisor.v1' \
  'supervisor_status=0' 'drain_acknowledged=1' 'child_exit_code=0' \
  'child_closed=1'; do
  grep -Fxq "$exact" "$logs/supervisor.stdout" ||
    fail "drain/exit evidence lost: $exact"
done
child_pid=$(sed -n 's/^child_pid=//p' "$logs/supervisor.stdout")
[[ $child_pid =~ ^[1-9][0-9]*$ ]] || fail 'supervised child PID is malformed'
kill -0 "$child_pid" 2>/dev/null && fail 'runtime owner survived drain'

"$nvidia_smi" --query-gpu=index,uuid,name,compute_cap,memory.used \
  --format=csv,noheader,nounits >"$logs/gpu.after" 2>"$logs/gpu.after.stderr"
[[ ! -s $logs/gpu.after.stderr ]] || fail 'final GPU inventory emitted stderr'
cmp -s "$logs/gpu.before" "$logs/gpu.after" ||
  fail 'GPU inventory or framebuffer use changed across serving'
"$nvidia_smi" --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits >"$logs/processes.after" \
  2>"$logs/processes.after.stderr"
[[ ! -s $logs/processes.after && ! -s $logs/processes.after.stderr ]] ||
  fail 'GPU compute process survived serving drain'
ps -eo pid=,args= >"$logs/process-inventory.raw.after" ||
  fail 'final process inventory failed'
awk -v owner="$$" '$1 != owner { print }' \
  "$logs/process-inventory.raw.after" >"$logs/process-inventory.after"
for exact_path in "$lunaflux" "$worker"; do
  if grep -F "$exact_path" "$logs/process-inventory.after" \
    >"$logs/process-survivor.$(basename -- "$exact_path")"; then
    fail 'runtime owner or worker process survived campaign'
  else
    grep_status=$?
    [[ $grep_status -eq 1 ]] || fail 'final process search failed'
  fi
done

lunaflux_prepare_evidence_manifest "$stage" || fail 'evidence inventory failed'
files_sha=$lunaflux_evidence_manifest_sha256
cat >"$stage/RESULT.txt" <<EOF
schema=lunaflux-qwen3-bf16-v12-serving-physical-result.v1
outcome=passed-native-framed-c1-correctness
runtime_recipe=dense_qwen3_bf16_paged_aot_v12
model_family=qwen3
model_layers=28
model_content_sha256=$model_content_sha
model_plan_sha256=$model_plan_sha
launch_sha256=$launch_sha
lunaflux_executable_sha256=$lunaflux_sha
worker_executable_sha256=$worker_sha
serving_client_sha256=$client_sha
supervisor_sha256=$supervisor_sha
reference_corpus_sha256=$corpus_sha
nvidia_smi_sha256=$nvidia_smi_sha
curl_sha256=$curl_sha
metrics_sha256=$metrics_sha
sampling_runtime=host
native_framed_persistent_serving=passed
greedy_corpus_match=passed
network_balance=accepts1,disconnects1
request_balance=queue0,active0
kv_balance=used0,free$total_pages
drain_acknowledged=1
owner_closed=1
worker_closed=1
gpu_processes_before=0
gpu_processes_during=1
gpu_processes_after=0
engine_servers_resident_concurrently=1
openai_sse_benchmark_phase=blocked
openai_sse_benchmark_blocker=authenticated-qwen-token-id-sse-benchmark-bridge-unavailable
standard_openai_responses_profile_satisfies_benchmark_adapter=false
benchmark_capacity=c1-correctness-only
release_bind_max_batch_rows=1
release_bind_max_query_rows=1
benchmark_c32=not-run-separate-release-profile
benchmark_c32_software_admission=present-not-exercised
benchmark_c32_release_bind_max_batch_rows=32
benchmark_c32_release_bind_max_query_rows=32
benchmark_performance_claim=not-made
evidence_files_manifest_sha256=$files_sha
EOF
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
published=1
trap - EXIT HUP INT TERM
printf 'outcome=qwen3-bf16-v12-native-serving-physical-published evidence=%s outer_seal_sha256=%s openai_sse=blocked benchmark_c32=not-run-separate-release-profile\n' \
  "$output" "$outer_sha"
