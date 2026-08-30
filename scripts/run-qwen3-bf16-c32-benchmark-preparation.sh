#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
TZ=UTC
export LC_ALL TZ
umask 077

fail() {
  printf 'Qwen3 c32 benchmark preparation rejected: %s\n' "$1" >&2
  exit 1
}

usage() {
  printf '%s\n' \
    "usage: $0 ABSOLUTE_NVCC ABSOLUTE_TOOLCHAIN_MANIFEST#sha256=HEX ABSOLUTE_SOURCE_MODEL_ROOT ABSOLUTE_SOURCE_MODEL_INVENTORY#sha256=HEX ABSOLUTE_MODEL_ADMISSION#sha256=HEX CONFIG_SHA256 MODEL_CONTENT_SHA256 TOKENIZER_SHA256 SOURCE_SAFETENSORS_LOCATOR SOURCE_SAFETENSORS_SHA256 ABSOLUTE_REFERENCE_CORPUS#sha256=HEX ABSOLUTE_RUNTIME#sha256=HEX ABSOLUTE_SUPERVISOR#sha256=HEX ABSOLUTE_TOKEN_ID_BRIDGE#sha256=HEX ABSOLUTE_WORKER#sha256=HEX ABSOLUTE_NVIDIA_SMI#sha256=HEX ABSOLUTE_CURL#sha256=HEX LUNAFLUX_REVISION_SHA256 EXPECTED_GPU_UUID EXPECTED_GPU_PCI RUNTIME_PORT BRIDGE_PORT ABSOLUTE_NEW_ARTIFACT_ROOT ABSOLUTE_NEW_EVIDENCE_ROOT" >&2
  exit 2
}

[[ $# -eq 24 ]] || usage
nvcc=$1
toolchain_argument=$2
source_model_root=$3
source_inventory_argument=$4
model_admission_argument=$5
config_sha=$6
model_content_sha=$7
tokenizer_sha=$8
source_locator=$9
shift 9
source_sha=$1
corpus_argument=$2
runtime_argument=$3
supervisor_argument=$4
bridge_argument=$5
worker_argument=$6
nvidia_smi_argument=$7
curl_argument=$8
revision_sha=$9
shift 9
expected_uuid=$1
expected_pci=$2
runtime_port=$3
bridge_port=$4
artifact_root=$5
evidence_output=$6

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"
. "$repo_root/scripts/immutable-evidence-directory.sh"
. "$repo_root/scripts/qwen3-c32-preparation-common.sh"

[[ $nvcc == /* && -f $nvcc && ! -L $nvcc && -x $nvcc &&
   $(realpath -- "$nvcc") == "$nvcc" ]] || fail 'NVCC is not canonical'
require_pinned_file "$toolchain_argument" 'CUDA toolchain manifest'
toolchain_manifest=$pinned_path
toolchain_sha=$pinned_sha
[[ -d $source_model_root && ! -L $source_model_root &&
   $(realpath -- "$source_model_root") == "$source_model_root" ]] ||
  fail 'source Qwen model root is not canonical'
require_pinned_file "$source_inventory_argument" 'source model inventory'
source_inventory=$pinned_path
source_inventory_sha=$pinned_sha
require_pinned_file "$model_admission_argument" 'model admission receipt'
model_admission=$pinned_path
model_admission_sha=$pinned_sha
for digest in "$config_sha" "$model_content_sha" "$tokenizer_sha" "$source_sha" \
  "$revision_sha"; do
  is_sha256 "$digest" || fail 'a Qwen/revision digest is malformed'
done
[[ $source_locator =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] ||
  fail 'source safetensors locator must be a direct model-root child'
source_safetensors=$source_model_root/$source_locator
[[ -f $source_safetensors && ! -L $source_safetensors ]] ||
  fail 'source safetensors is absent'
[[ $(sha256_file "$source_safetensors") == "$source_sha" ]] ||
  fail 'source safetensors digest differs'
[[ $(sha256_file "$source_model_root/config.json") == "$config_sha" ]] ||
  fail 'Qwen config digest differs'
[[ $(sha256_file "$source_model_root/tokenizer.json") == "$tokenizer_sha" ]] ||
  fail 'Qwen tokenizer digest differs'
require_pinned_file "$corpus_argument" 'Qwen reference corpus'
corpus=$pinned_path
corpus_sha=$pinned_sha
require_executable "$runtime_argument" 'native runtime'
runtime=$pinned_path
runtime_sha=$pinned_sha
require_executable "$supervisor_argument" 'native supervisor'
supervisor=$pinned_path
supervisor_sha=$pinned_sha
require_executable "$bridge_argument" 'token-ID bridge'
bridge=$pinned_path
bridge_sha=$pinned_sha
require_executable "$worker_argument" 'device worker'
worker=$pinned_path
worker_sha=$pinned_sha
require_executable "$nvidia_smi_argument" 'nvidia-smi'
nvidia_smi=$pinned_path
nvidia_smi_sha=$pinned_sha
require_executable "$curl_argument" 'curl'
curl=$pinned_path
curl_sha=$pinned_sha
[[ $expected_uuid =~ ^GPU-[0-9A-Fa-f-]{36}$ ]] || fail 'GPU UUID is malformed'
[[ $expected_pci =~ ^[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$ ]] ||
  fail 'GPU PCI identity is malformed'
for port in "$runtime_port" "$bridge_port"; do
  [[ $port =~ ^[0-9]+$ && $port -ge 1024 && $port -le 65535 ]] ||
    fail 'a loopback port is invalid'
done
[[ $runtime_port != "$bridge_port" ]] || fail 'runtime and bridge ports must differ'
require_new_output "$artifact_root" 'artifact root'
require_new_output "$evidence_output" 'evidence output'
command -v setsid >/dev/null 2>&1 || fail 'setsid is unavailable'

python3 -B "$repo_root/benchmarks/qwen3_comparison/verify_model_inventory.py" \
  "$source_model_root" "$source_inventory" "$source_inventory_sha" ||
  fail 'source Qwen inventory differs'
python3 -B "$repo_root/benchmarks/qwen3_comparison/verify_model_admission.py" \
  "$source_model_root" "$model_admission_argument" ||
  fail 'model admission differs'
grep -Fq "\"model_content_sha256\":\"$model_content_sha\"" "$model_admission" ||
  fail 'model admission content digest differs'
grep -Fq "\"config_sha256\":\"$config_sha\"" "$model_admission" ||
  fail 'model admission config digest differs'
grep -Fq "\"tokenizer_json_sha256\":\"$tokenizer_sha\"" "$model_admission" ||
  fail 'model admission tokenizer digest differs'
grep -Fq "\"source_model_inventory_sha256\":\"$source_inventory_sha\"" "$model_admission" ||
  fail 'model admission inventory digest differs'
grep -Fq "\"model_root\":\"$source_model_root\"" "$model_admission" ||
  fail 'model admission source root differs'
grep -Fq "\"source_model_inventory\":\"$source_inventory\"" "$model_admission" ||
  fail 'model admission inventory path differs'

artifact_created=0
evidence_parent=${evidence_output%/*}
evidence_name=${evidence_output##*/}
stage=$(mktemp -d "$evidence_parent/.${evidence_name}.partial.XXXXXX")
stage=$(realpath -- "$stage")
logs=$stage/logs
mkdir "$logs"
server_pid=
server_pgid=
published=0

cleanup() {
  stop_server || true
  if [[ $published -ne 1 ]]; then
    if [[ -d ${stage:-} ]]; then
      chmod -R u+rwX "$stage" 2>/dev/null || true
      rm -rf -- "$stage"
    fi
    if [[ $artifact_created -eq 1 && -d $artifact_root ]]; then
      chmod -R u+rwX "$artifact_root" 2>/dev/null || true
      rm -rf -- "$artifact_root"
    fi
  fi
}
trap cleanup EXIT HUP INT TERM

mkdir "$artifact_root"
artifact_created=1
model_root=$artifact_root/model-input
mkdir "$model_root"
cp "$source_model_root/config.json" "$model_root/config.json"
cp "$source_model_root/tokenizer.json" "$model_root/tokenizer.json"
cp "$source_safetensors" "$model_root/$source_locator"

driver_report=$logs/cuda-driver.v1
"$repo_root/scripts/inspect-luna-cuda-aot-driver.sh" "$nvcc" >"$driver_report"
driver_identity=$(field driver_identity_sha256 "$driver_report")
compiler_version=$(field compiler_version "$driver_report")
[[ $compiler_version == 13.1.115 ]] || fail 'CUDA compiler must be exactly 13.1.115'
[[ $(field driver_identity_sha256 "$toolchain_manifest") == "$driver_identity" ]] ||
  fail 'CUDA driver is not admitted by the exact toolchain manifest'
gpu_identity=$("$nvidia_smi" --id=0 --query-gpu=uuid,pci.bus_id \
  --format=csv,noheader,nounits)
[[ $gpu_identity == "$expected_uuid, $expected_pci" ]] ||
  fail 'GPU 0 UUID/PCI identity differs'
printf '%s\n' "$gpu_identity" >"$logs/device-identity.txt"
"$nvidia_smi" --id=0 --query-compute-apps=pid \
  --format=csv,noheader,nounits >"$logs/processes.before" 2>"$logs/processes.before.stderr"
[[ ! -s $logs/processes.before && ! -s $logs/processes.before.stderr ]] ||
  fail 'target GPU is not idle before c32 preparation'

moon build cmd/lunaflux_qwen3_weight_convert \
  cmd/lunaflux_qwen3_bf16_candidate_export \
  cmd/lunaflux_qwen3_bf16_release_bind \
  --target native --release --deny-warn --warn-list +73 \
  >"$logs/moon-build.stdout" 2>"$logs/moon-build.stderr" ||
  fail 'Qwen c32 preparation CLIs did not build'
# Moon writes cache/progress diagnostics to stderr on current Linux releases,
# and native dependencies can emit compiler diagnostics without failing the
# warning-denied MoonBit build. Preserve that stream for diagnosis; the build
# status plus --deny-warn is the admission boundary.
converter=$repo_root/_build/native/release/build/cmd/lunaflux_qwen3_weight_convert/lunaflux_qwen3_weight_convert.exe
exporter=$repo_root/_build/native/release/build/cmd/lunaflux_qwen3_bf16_candidate_export/lunaflux_qwen3_bf16_candidate_export.exe
binder=$repo_root/_build/native/release/build/cmd/lunaflux_qwen3_bf16_release_bind/lunaflux_qwen3_bf16_release_bind.exe
for executable in "$converter" "$exporter" "$binder"; do
  [[ -x $executable && -f $executable && ! -L $executable ]] ||
    fail 'a Qwen c32 preparation CLI is absent'
done

numeric_locator=qwen3-0.6b-bf16-c32.numeric
"$converter" "$model_root" config.json "$config_sha" "$model_content_sha" \
  "$source_locator" "$source_sha" 32 "$numeric_locator" \
  >"$logs/weight-convert.stdout" 2>"$logs/weight-convert.stderr" ||
  fail 'Qwen BF16 c32 numeric conversion failed'
[[ ! -s $logs/weight-convert.stderr ]] || fail 'numeric conversion emitted stderr'
[[ $(field max_batch_rows "$logs/weight-convert.stdout") == 32 ]] ||
  fail 'numeric conversion did not bind c32'
numeric_sha=$(field output_artifact_sha256 "$logs/weight-convert.stdout")
route_sha=$(field route_manifest_sha256 "$logs/weight-convert.stdout")
is_sha256 "$numeric_sha" && is_sha256 "$route_sha" ||
  fail 'numeric conversion emitted malformed identities'
[[ $(sha256_file "$model_root/$numeric_locator") == "$numeric_sha" ]] ||
  fail 'numeric artifact digest differs after publication'

candidate=$artifact_root/candidate
"$exporter" "$model_root" config.json "$config_sha" "$model_content_sha" \
  "$numeric_locator" "$numeric_sha" "$route_sha" "$toolchain_sha" \
  13 1 115 8 8192 32 256 5120 "$candidate" \
  >"$logs/candidate-export.stdout" 2>"$logs/candidate-export.stderr" ||
  fail 'Qwen c32 candidate export failed'
[[ ! -s $logs/candidate-export.stderr ]] || fail 'candidate export emitted stderr'
[[ $(field max_query_rows "$logs/candidate-export.stdout") == 32 &&
   $(field max_query_tokens "$logs/candidate-export.stdout") == 256 ]] ||
  fail 'candidate export geometry is not production c32'
candidate_inventory=$candidate/candidate.files.sha256
candidate_inventory_sha=$(sha256_file "$candidate_inventory")

compiled=$artifact_root/compiled
"$repo_root/scripts/build-luna-bf16-kernel-set.sh" "$nvcc" \
  "$toolchain_argument" "$candidate/candidate-root" \
  "$candidate_inventory#sha256=$candidate_inventory_sha" "$compiled" \
  >"$logs/kernel-build.stdout" 2>"$logs/kernel-build.stderr" ||
  fail 'Qwen c32 AOT kernel build failed'
[[ ! -s $logs/kernel-build.stderr ]] || fail 'AOT kernel build emitted stderr'
"$repo_root/scripts/verify-luna-bf16-kernel-set.sh" "$compiled" \
  >"$logs/kernel-verify.stdout" 2>"$logs/kernel-verify.stderr" ||
  fail 'Qwen c32 compiled set failed verification'
[[ ! -s $logs/kernel-verify.stderr ]] || fail 'compiled-set verification emitted stderr'

release=$artifact_root/release
"$binder" "$model_root" config.json "$config_sha" "$model_content_sha" \
  "$numeric_locator" "$numeric_sha" "$route_sha" "$compiled" \
  "$toolchain_manifest" "$toolchain_sha" 13 1 115 8 8192 32 256 5120 \
  "$release" >"$artifact_root/release-bind.stdout" \
  2>"$logs/release-bind.stderr" || fail 'Qwen c32 release bind failed'
[[ ! -s $logs/release-bind.stderr ]] || fail 'release bind emitted stderr'
release_bind=$artifact_root/release-bind.stdout
release_bind_sha=$(sha256_file "$release_bind")
for exact in 'max_batch_rows=32' 'max_query_rows=32' 'max_query_tokens=256' \
  'tokens_per_page=8' 'total_page_count=8192' 'max_page_table_entries=5120' \
  'target=sm_120' 'compiler_invoked=0' 'device_opened=0'; do
  grep -Fxq "$exact" "$release_bind" || fail "release bind lost: $exact"
done
model_plan_sha=$(field model_plan_sha256 "$release_bind")

deployment=$artifact_root/deployment
"$repo_root/scripts/materialize-qwen3-bf16-v12-launch.sh" \
  "$model_root" "$config_sha" "$model_content_sha" "$tokenizer_sha" \
  "$numeric_locator" "$numeric_sha" "$route_sha" "$release" \
  "$release_bind#sha256=$release_bind_sha" "$worker_argument" \
  "$corpus_argument" "$runtime_port" "$deployment" \
  native-framed-c32-benchmark-v1 \
  >"$logs/deployment-materialize.stdout" 2>"$logs/deployment-materialize.stderr" ||
  fail 'Qwen c32 deployment materialization failed'
[[ ! -s $logs/deployment-materialize.stderr ]] ||
  fail 'deployment materialization emitted stderr'
launch=$deployment/lunaflux.launch.json
launch_sha=$(sha256_file "$launch")
grep -Fq '"max_batch_rows":32' "$deployment/model-root/runtime/descriptor.json" ||
  fail 'deployment descriptor is not c32'
grep -Fq '"max_active_requests":32' "$deployment/policy-root/instance/policy.json" ||
  fail 'deployment policy is not c32'

capacity=$artifact_root/authenticated-capacity.json
"$repo_root/scripts/materialize-qwen3-authenticated-capacity.sh" \
  "$release_bind#sha256=$release_bind_sha" "$deployment#sha256=$launch_sha" \
  "$runtime_argument" "$bridge_argument" "$capacity" \
  >"$logs/capacity-materialize.stdout" 2>"$logs/capacity-materialize.stderr" ||
  fail 'authenticated c32 capacity materialization failed'
[[ ! -s $logs/capacity-materialize.stderr ]] ||
  fail 'capacity materialization emitted stderr'
capacity_sha=$(sha256_file "$capacity")
"$repo_root/scripts/verify-qwen3-authenticated-capacity.sh" \
  "$release_bind#sha256=$release_bind_sha" "$deployment#sha256=$launch_sha" \
  "$runtime_argument" "$bridge_argument" "$capacity#sha256=$capacity_sha" \
  >"$logs/capacity-verify.stdout" 2>"$logs/capacity-verify.stderr" ||
  fail 'authenticated c32 capacity verification failed'
[[ ! -s $logs/capacity-verify.stderr ]] || fail 'capacity verifier emitted stderr'
grep -Fxq 'max_concurrency=32' "$logs/capacity-verify.stdout" ||
  fail 'capacity verifier did not admit concurrency 32'

request=$stage/request.json
printf '%s\n' '{"schema":"lunaflux.benchmark-token-ids-request.v1","model":"Qwen3-0.6B","input_token_ids":[151644,872,198,840,20772,304,825,2805,11652,3170,72349,44378,12850,13,151645,198,151644,77091,198,151667,271,151668,271],"max_output_tokens":2,"sampling":{"mode":"greedy","temperature":0,"top_p":1,"seed":0,"ignore_eos":true},"stream":true}' >"$request"

runtime_gpu_identity=$("$nvidia_smi" --id=0 --query-gpu=uuid,pci.bus_id \
  --format=csv,noheader,nounits)
[[ $runtime_gpu_identity == "$expected_uuid, $expected_pci" ]] ||
  fail 'GPU 0 identity changed before native runtime admission'
printf '%s\n' "$runtime_gpu_identity" >"$logs/device-identity.runtime.txt"
export CUDA_VISIBLE_DEVICES="$expected_uuid"
setsid "$repo_root/scripts/start-qwen3-lunaflux-benchmark-server.sh" \
  native "$revision_sha" "$source_model_root" "$model_admission_argument" \
  127.0.0.1 "$bridge_port" "$runtime_argument" "$supervisor_argument" \
  "$bridge_argument" "$deployment#sha256=$launch_sha" \
  "$source_model_root/tokenizer.json#sha256=$tokenizer_sha" \
  "$launch#sha256=$launch_sha" "$release_bind#sha256=$release_bind_sha" \
  "$capacity#sha256=$capacity_sha" "127.0.0.1:$runtime_port" \
  "$model_content_sha" "$model_plan_sha" 4096 256 151935 40960 \
  >"$logs/server.stdout" 2>"$logs/server.stderr" &
server_pid=$!
server_pgid=$server_pid

ready=0
for _ in $(seq 1 6000); do
  if "$curl" --fail --silent --show-error --max-time 2 \
    "http://127.0.0.1:$bridge_port/health" -o "$logs/bridge-health.json" \
    2>"$logs/bridge-health.stderr"; then
    ready=1
    break
  fi
  kill -0 "$server_pid" 2>/dev/null || break
  sleep 0.1
done
[[ $ready -eq 1 ]] || fail 'Qwen c32 native runtime/bridge did not become ready'
grep -Fxq '{"status":"healthy","model":"Qwen3-0.6B"}' "$logs/bridge-health.json" ||
  fail 'token-ID bridge health identity differs'

smoke_failed=0
smoke_pids=()
for request_id in $(seq 1 32); do
  "$curl" --fail --silent --show-error --max-time 120 \
    -H 'Content-Type: application/json' -H 'Accept: text/event-stream' \
    --data-binary "@$request" \
    "http://127.0.0.1:$bridge_port/benchmark/v1/token-ids" \
    -o "$logs/smoke.$request_id.sse" \
    2>"$logs/smoke.$request_id.stderr" &
  smoke_pids+=("$!")
done
for smoke_pid in "${smoke_pids[@]}"; do
  wait "$smoke_pid" || smoke_failed=1
done
[[ $smoke_failed -eq 0 ]] || fail 'one or more c32 bridge requests failed'
for request_id in $(seq 1 32); do
  response=$logs/smoke.$request_id.sse
  [[ ! -s $logs/smoke.$request_id.stderr ]] || fail 'a c32 smoke request emitted stderr'
  event_sequence=$(sed -n \
    -e 's/^data: {"schema":"lunaflux.benchmark-token.v1","token_id":\([0-9][0-9]*\),.*$/token:\1/p' \
    -e 's/^data: {"schema":"lunaflux.benchmark-terminal.v1",.*$/terminal/p' \
    -e 's/^data: \[DONE\]$/done/p' "$response" | paste -sd, -)
  [[ $event_sequence == 'token:92648,token:4532,terminal,done' ]] ||
    fail 'a c32 smoke response differs from the Qwen greedy prefix'
done
printf '%s\n' \
  'outcome=qwen3-c32-token-id-bridge-smoke-pass requests=32 output_tokens_per_request=2 first_token=92648 second_token=4532' \
  >"$logs/c32-smoke.stdout"

"$nvidia_smi" --id=0 --query-compute-apps=pid \
  --format=csv,noheader,nounits >"$logs/processes.during" 2>"$logs/processes.during.stderr"
[[ ! -s $logs/processes.during.stderr &&
   $(wc -l <"$logs/processes.during" | tr -d ' ') -eq 1 ]] ||
  fail 'exactly one Qwen GPU engine was not resident during the smoke'

drained_pgid=$server_pgid
kill -TERM "$server_pid"
drain_complete=0
for _ in $(seq 1 600); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    drain_complete=1
    break
  fi
  sleep 0.1
done
[[ $drain_complete -eq 1 ]] || fail 'Qwen c32 combined server drain timed out'
server_status=0
wait "$server_pid" || server_status=$?
server_pid=
server_pgid=
[[ $server_status -eq 0 ]] || fail 'Qwen c32 combined server did not drain cleanly'
if kill -0 -- "-$drained_pgid" 2>/dev/null; then
  fail 'Qwen c32 process group survived clean drain'
fi
[[ ! -s $logs/server.stderr ]] || fail 'Qwen c32 combined server emitted stderr'
for exact in 'readiness: true' 'schema=lunaflux-qwen3-native-supervisor.v1' \
  'supervisor_status=0' 'drain_acknowledged=1' 'child_exit_code=0' 'child_closed=1' \
  'schema=lunaflux.qwen3-token-id-sse-bridge.v1' 'max_connections=32'; do
  grep -Fq "$exact" "$logs/server.stdout" || fail "clean lifecycle evidence lost: $exact"
done
"$nvidia_smi" --id=0 --query-compute-apps=pid \
  --format=csv,noheader,nounits >"$logs/processes.after" 2>"$logs/processes.after.stderr"
[[ ! -s $logs/processes.after && ! -s $logs/processes.after.stderr ]] ||
  fail 'GPU process survived Qwen c32 drain'
ps -eo pid=,pgid=,args= >"$logs/process-inventory.after"

find "$artifact_root" -type l -print -quit | grep -q . &&
  fail 'artifact root contains a symbolic link'
artifact_manifest=$artifact_root/ARTIFACTS.sha256
: >"$artifact_manifest"
while IFS= read -r -d '' file; do
  relative=${file#"$artifact_root"/}
  [[ $file == "$artifact_manifest" ]] ||
    printf '%s  %s\n' "$(sha256_file "$file")" "$relative" >>"$artifact_manifest"
done < <(find "$artifact_root" -type f -print0 | LC_ALL=C sort -z)
artifact_manifest_sha=$(sha256_file "$artifact_manifest")
find "$artifact_root" -type f -exec chmod 0444 {} +
chmod 0555 "$artifact_root/deployment/bin/lunaflux-device-worker"
find "$artifact_root" -type d -exec chmod 0555 {} +

cat >"$stage/IDENTITIES.txt" <<EOF
schema=lunaflux-qwen3-bf16-c32-benchmark-preparation-identities.v1
model_content_sha256=$model_content_sha
source_model_inventory_sha256=$source_inventory_sha
model_admission_sha256=$model_admission_sha
config_sha256=$config_sha
tokenizer_sha256=$tokenizer_sha
source_safetensors_sha256=$source_sha
toolchain_sha256=$toolchain_sha
driver_identity_sha256=$driver_identity
runtime_sha256=$runtime_sha
supervisor_sha256=$supervisor_sha
token_id_bridge_sha256=$bridge_sha
worker_sha256=$worker_sha
nvidia_smi_sha256=$nvidia_smi_sha
curl_sha256=$curl_sha
revision_sha256=$revision_sha
gpu_uuid=$expected_uuid
gpu_pci=$expected_pci
numeric_sha256=$numeric_sha
weight_route_manifest_sha256=$route_sha
model_plan_sha256=$model_plan_sha
release_bind_sha256=$release_bind_sha
launch_sha256=$launch_sha
capacity_sha256=$capacity_sha
artifact_manifest_sha256=$artifact_manifest_sha
EOF

lunaflux_prepare_evidence_manifest "$stage" || fail 'evidence inventory failed'
cat >"$stage/RESULT.txt" <<EOF
schema=lunaflux-qwen3-bf16-c32-benchmark-preparation-result.v1
outcome=passed
model_family=qwen3
runtime_recipe=dense_qwen3_bf16_paged_aot_v12
target=sm_120
compiler_version=13.1.115
release_bind_max_batch_rows=32
release_bind_max_query_rows=32
release_bind_max_query_tokens=256
tokens_per_page=8
total_page_count=8192
max_page_table_entries=5120
authenticated_capacity=32
concurrent_correctness_requests=32
output_tokens_per_request=2
all_requests_matched_qwen_greedy_prefix=true
native_runtime_closed=true
native_supervisor_closed=true
token_id_bridge_closed=true
device_worker_closed=true
gpu_processes_before=0
gpu_processes_during=1
gpu_processes_after=0
authentication_work=request_path=0,startup_offline_only=1
benchmark_performance_claim=not-made
artifact_root=$artifact_root
artifact_manifest_sha256=$artifact_manifest_sha
evidence_files_manifest_sha256=$lunaflux_evidence_manifest_sha256
EOF
{
  printf '%s  FILES.sha256\n' "$(sha256_file "$stage/FILES.sha256")"
  printf '%s  RESULT.txt\n' "$(sha256_file "$stage/RESULT.txt")"
} >"$stage/OUTER_SEAL.sha256"
outer_sha=$(sha256_file "$stage/OUTER_SEAL.sha256")
lunaflux_seal_evidence_directory "$stage" || fail 'evidence sealing failed'
[[ ! -e $evidence_output && ! -L $evidence_output ]] ||
  fail 'evidence output appeared before publication'
chmod 0755 "$stage"
mv -- "$stage" "$evidence_output"
chmod 0555 "$evidence_output"
stage=
published=1
trap - EXIT HUP INT TERM
printf 'outcome=qwen3-bf16-c32-benchmark-preparation-published artifacts=%s artifact_manifest_sha256=%s evidence=%s outer_seal_sha256=%s performance_claim=not-made\n' \
  "$artifact_root" "$artifact_manifest_sha" "$evidence_output" "$outer_sha"
