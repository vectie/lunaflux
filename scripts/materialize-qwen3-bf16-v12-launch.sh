#!/bin/sh

set -eu
LC_ALL=C
TZ=UTC
export LC_ALL TZ
umask 077

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/release-bundle-common.sh"

fail() {
  printf '%s\n' "LunaFlux Qwen3 v12 launch materialization rejected: $1" >&2
  exit 1
}

usage() {
  printf '%s\n' \
    'usage: materialize-qwen3-bf16-v12-launch.sh ABSOLUTE_MODEL_ROOT CONFIG_SHA256 MODEL_CONTENT_SHA256 TOKENIZER_SHA256 NUMERIC_LOCATOR NUMERIC_SHA256 ROUTE_SHA256 ABSOLUTE_RELEASE_BIND_ROOT ABSOLUTE_RELEASE_BIND_STDOUT#sha256=HEX ABSOLUTE_WORKER#sha256=HEX ABSOLUTE_REFERENCE_CORPUS#sha256=HEX LISTEN_PORT ABSOLUTE_NEW_OUTPUT [native-framed-v1|native-framed-c32-benchmark-v1|openai-responses-v1] [ABSOLUTE_REUSABLE_FUSED_RESIDUAL_RUNTIME#sha256=HEX]' >&2
  exit 2
}

case "$#" in 13|14|15) ;; *) usage ;; esac
model_root=$1
config_sha=$2
model_content_sha=$3
tokenizer_sha=$4
numeric_locator=$5
numeric_sha=$6
route_sha=$7
release_root=$8
bind_argument=$9
worker_argument=${10}
corpus_argument=${11}
listen_port=${12}
output=${13}
materialization_profile=${14:-native-framed-v1}
fused_residual_argument=${15:-}
case "$materialization_profile" in
  native-framed-v1|native-framed-c32-benchmark-v1|openai-responses-v1) ;;
  *) fail 'materialization profile is unsupported' ;;
esac
fused_runtime_json=
descriptor_schema=lunaflux.runtime.qwen3_bf16.v1
if [ -n "$fused_residual_argument" ]; then
  case "$fused_residual_argument" in /*#sha256=*) ;; *) usage ;; esac
  fused_residual=${fused_residual_argument%#sha256=*}
  fused_residual_sha=${fused_residual_argument##*#sha256=}
  bundle_require_canonical_absolute_file "$fused_residual"
  bundle_require_digest "$fused_residual" "$fused_residual_sha" \
    'reusable fused residual runtime'
  [ "$(sed -n '1p' "$fused_residual")" = \
    'schema=lunaflux-reusable-fused-residual-rmsnorm-runtime.v1' ] ||
    fail 'reusable fused residual runtime schema is invalid'
  descriptor_schema=lunaflux.runtime.qwen3_bf16.v2
  fused_runtime_json=',"fused_runtime":{"locator":"reusable-fused-residual.runtime.v1","sha256":"'"$fused_residual_sha"'"}'
fi

case "$bind_argument" in /*#sha256=*) ;; *) usage ;; esac
bind_stdout=${bind_argument%#sha256=*}
bind_stdout_sha=${bind_argument##*#sha256=}
case "$worker_argument" in /*#sha256=*) ;; *) usage ;; esac
worker=${worker_argument%#sha256=*}
worker_sha=${worker_argument##*#sha256=}
case "$corpus_argument" in /*#sha256=*) ;; *) usage ;; esac
corpus=${corpus_argument%#sha256=*}
corpus_sha=${corpus_argument##*#sha256=}

bundle_require_canonical_absolute_directory "$model_root"
bundle_require_canonical_absolute_directory "$release_root"
for file in "$bind_stdout" "$worker" "$corpus"; do
  bundle_require_canonical_absolute_file "$file"
done
bundle_require_digest "$bind_stdout" "$bind_stdout_sha" 'release-bind stdout'
bundle_require_digest "$worker" "$worker_sha" 'worker executable'
bundle_require_digest "$corpus" "$corpus_sha" 'reference corpus'
[ -x "$worker" ] || fail 'worker executable is not executable'
for digest in "$config_sha" "$model_content_sha" "$tokenizer_sha" \
  "$numeric_sha" "$route_sha"; do
  bundle_is_lower_sha256 "$digest" || fail 'an authority digest is invalid'
done
bundle_is_strict_relative "$numeric_locator" || fail 'numeric locator is invalid'
case "$numeric_locator" in */*) fail 'numeric artifact must be a direct model-root child' ;; esac
case "$listen_port" in ''|*[!0-9]*) fail 'listen port is invalid' ;; esac
[ "$listen_port" -ge 1 ] && [ "$listen_port" -le 65535 ] ||
  fail 'listen port is outside 1..65535'

config=$model_root/config.json
tokenizer=$model_root/tokenizer.json
numeric=$model_root/$numeric_locator
for file in "$config" "$tokenizer" "$numeric"; do
  bundle_require_canonical_absolute_file "$file"
done
bundle_require_digest "$config" "$config_sha" 'Qwen config'
bundle_require_digest "$tokenizer" "$tokenizer_sha" 'Qwen tokenizer'
bundle_require_digest "$numeric" "$numeric_sha" 'Qwen numeric weights'

bind_value() {
  bv=$(sed -n "s/^$1=//p" "$bind_stdout")
  [ -n "$bv" ] && [ "$(grep -c "^$1=" "$bind_stdout")" -eq 1 ] ||
    fail "release-bind field is absent or duplicated: $1"
  printf '%s\n' "$bv"
}
[ "$(bind_value schema)" = lunaflux-qwen3-bf16-release-bind.v1 ] ||
  fail 'release-bind schema is unsupported'
[ "$(bind_value recipe)" = dense_qwen3_bf16_paged_aot_v12 ] ||
  fail 'release-bind recipe is not Qwen v12'
[ "$(bind_value model_content_sha256)" = "$model_content_sha" ] ||
  fail 'release-bind model content differs'
[ "$(bind_value weight_route_manifest_sha256)" = "$route_sha" ] ||
  fail 'release-bind weight route differs'
[ "$(bind_value target)" = sm_120 ] || fail 'release-bind target is not sm120'
[ "$(bind_value compiler_invoked)" = 0 ] &&
  [ "$(bind_value device_opened)" = 0 ] &&
  [ "$(bind_value runtime_authority)" = 0 ] ||
  fail 'release binder crossed its offline boundary'
sampling_runtime_json=
sampling_runtime=host
authenticated_embedded_greedy_sampling=$(
  bind_value authenticated_embedded_greedy_sampling
)
case "$materialization_profile" in
  native-framed-c32-benchmark-v1)
    [ "$authenticated_embedded_greedy_sampling" = 1 ] ||
      fail 'Qwen c32 benchmark requires authenticated device greedy sampling'
    sampling_runtime_json=',"sampling_runtime":"embedded_cuda_greedy_v1"'
    sampling_runtime=embedded_cuda_greedy_v1
    ;;
  native-framed-v1|openai-responses-v1)
    [ "$authenticated_embedded_greedy_sampling" = 0 ] ||
      fail 'Qwen host-sampling profile admitted device greedy sampling'
    ;;
esac

model_plan_sha=$(bind_value model_plan_sha256)
manifest_relative=$(bind_value kernel_manifest_relative)
manifest_sha=$(bind_value kernel_manifest_sha256)
bootstrap_sha=$(bind_value admitted_bootstrap_sha256)
bootstrap_source_sha=$(bind_value bootstrap_source_sha256)
inventory_sha=$(bind_value kernel_inventory_sha256)
kernel_plan_sha=$(bind_value kernel_root_plan_sha256)
tokens_per_page=$(bind_value tokens_per_page)
total_page_count=$(bind_value total_page_count)
max_sequence_tokens=$(bind_value max_sequence_tokens)
max_page_table_entries=$(bind_value max_page_table_entries)
for digest in "$model_plan_sha" "$manifest_sha" "$bootstrap_sha" \
  "$bootstrap_source_sha" "$inventory_sha" "$kernel_plan_sha"; do
  bundle_is_lower_sha256 "$digest" || fail 'release-bind emitted an invalid digest'
done
[ "$manifest_relative" = lunaflux.execution.json ] ||
  fail 'release-bind manifest locator differs'
[ "$tokens_per_page" -gt 0 ] && [ "$total_page_count" -gt 0 ] &&
  [ "$max_sequence_tokens" -gt 0 ] && [ "$max_page_table_entries" -gt 0 ] ||
  fail 'release-bind geometry is invalid'
context_page_table_entries=$((
  (max_sequence_tokens + tokens_per_page - 1) / tokens_per_page
))
[ "$context_page_table_entries" -le "$max_page_table_entries" ] &&
  [ "$max_page_table_entries" -le "$total_page_count" ] ||
  fail 'release-bind page-table geometry is inconsistent'

policy_schema=lunaflux.instance-policy.v1
external_protocol_json=
openai_service_json=
max_batch_rows=1
max_prefill_rows=1
max_decode_rows=1
max_plan_rows=1
max_plan_tokens=1
max_plan_pages=$max_page_table_entries
max_completion_slots=1
scheduler_step_token_budget=1
scheduler_max_active_requests=1
scheduler_max_waiting_requests=1
scheduler_prefill_chunk_tokens=1
scheduler_output_event_capacity=512
preparation_lane_count=1
preparation_storage_int_cells=2000000
preparation_storage_byte_cells=8388608
tokenizer_max_token_bytes=4096
max_decoded_delta_bytes=$((tokenizer_max_token_bytes + 3))
case "$materialization_profile" in
  native-framed-v1)
    [ "$(bind_value max_batch_rows)" -eq 1 ] &&
      [ "$(bind_value max_query_rows)" -eq 1 ] &&
      [ "$(bind_value max_query_tokens)" -eq 1 ] ||
      fail 'native correctness profile requires authenticated c1 release geometry'
    ;;
  native-framed-c32-benchmark-v1|openai-responses-v1)
    release_max_batch_rows=$(bind_value max_batch_rows)
    release_max_query_rows=$(bind_value max_query_rows)
    release_max_query_tokens=$(bind_value max_query_tokens)
    [ "$release_max_batch_rows" -eq 32 ] &&
      [ "$release_max_query_rows" -eq 32 ] &&
      [ "$release_max_query_tokens" -ge 32 ] ||
      fail 'benchmark profile requires authenticated c32 release geometry'
    max_batch_rows=$release_max_batch_rows
    max_prefill_rows=$release_max_query_rows
    max_decode_rows=$release_max_query_rows
    max_plan_rows=$release_max_query_rows
    max_plan_tokens=$release_max_query_tokens
    max_plan_pages=$max_page_table_entries
    max_completion_slots=$release_max_query_rows
    scheduler_step_token_budget=$release_max_query_tokens
    scheduler_max_active_requests=$release_max_batch_rows
    scheduler_max_waiting_requests=$release_max_batch_rows
    scheduler_prefill_chunk_tokens=$release_max_query_tokens
    scheduler_output_event_capacity=16384
    preparation_lane_count=$((
      scheduler_max_active_requests + scheduler_max_waiting_requests
    ))
    preparation_storage_int_cells=38282560
    preparation_storage_byte_cells=80499072
    ;;
esac
case "$materialization_profile" in
  openai-responses-v1)
    policy_schema=lunaflux.instance-policy.v3
    external_protocol_json=',"external_protocol":{"mode":"openai_responses_v1","transport_security":"loopback_plaintext","control_authentication":"deployment_bearer","invocation_path":"/v1/responses","health_path":"/healthz","readiness_path":"/readyz","drain_path":"/v1/drain","drain_method":"POST"}'
    openai_service_json=',"openai_service":{"maximum_credential_bytes":128,"max_head_bytes":65536,"max_body_bytes":1048576,"http_step_work_units":256,"response_step_work_units":256,"inbound_step_work_units":256,"outbound_step_work_units":256,"max_messages":32,"system_prefix":"<|im_start|>system\\n","system_suffix":"<|im_end|>\\n","user_prefix":"<|im_start|>user\\n","user_suffix":"<|im_end|>\\n","assistant_prefix":"<|im_start|>assistant\\n","assistant_suffix":"<|im_end|>\\n","assistant_cue":"<|im_start|>assistant\\n","max_rendered_prompt_bytes":65536,"model_alias":"qwen3-0.6b-bf16","response_id_prefix":"resp_lunaflux_qwen3_","cache_scope_ascii":"qwen3-openai-v1","max_new_tokens":256,"context_ceiling":'"$max_sequence_tokens"',"sampling_seed":1,"deadline_milliseconds":60000}'
    ;;
esac

plan=$release_root/kernel-root.plan.v1
inventory=$release_root/kernel.files.sha256
bootstrap_source_receipt=$release_root/bootstrap-source.v12.sha256
payload=$release_root/payload
bundle_require_canonical_absolute_file "$plan"
bundle_require_canonical_absolute_file "$inventory"
bundle_require_canonical_absolute_file "$bootstrap_source_receipt"
bundle_require_canonical_absolute_directory "$payload"
bundle_require_digest "$plan" "$kernel_plan_sha" 'kernel-root plan'
bundle_require_digest "$inventory" "$inventory_sha" 'kernel inventory'
bundle_require_digest "$payload/$manifest_relative" "$manifest_sha" 'execution manifest'
[ "$(wc -l <"$bootstrap_source_receipt" | tr -d ' ')" -eq 1 ] ||
  fail 'bootstrap-source receipt must contain exactly one line'
[ "$(sed -n '1p' "$bootstrap_source_receipt")" = "$bootstrap_source_sha" ] ||
  fail 'bootstrap-source receipt differs from release binding'

case "$output" in /*) ;; *) fail 'output path is not absolute' ;; esac
case "$output" in /|*//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
  fail 'output path is not a safe canonical absolute path'
  ;;
esac
[ ! -e "$output" ] && [ ! -L "$output" ] || fail 'output already exists'
output_parent=$(CDPATH= cd -- "$(dirname -- "$output")" && pwd -P)
[ "$output_parent/$(basename -- "$output")" = "$output" ] ||
  fail 'output parent is not canonical'

scratch=$(mktemp -d /tmp/lunaflux-qwen3-v12-launch.XXXXXX) ||
  fail 'could not create private scratch'
complete=0
output_created=0
cleanup() {
  chmod -R u+w "$scratch" 2>/dev/null || true
  rm -rf -- "$scratch"
  if [ "$complete" -ne 1 ] && [ "$output_created" -eq 1 ] &&
    [ -d "$output" ]; then
    chmod -R u+w "$output" 2>/dev/null || true
    rm -rf -- "$output"
  fi
}
trap cleanup EXIT HUP INT TERM

kernel_source=$scratch/kernel-source
mkdir "$kernel_source"
cp "$plan" "$kernel_source/kernel-root.plan.v1"
cp "$inventory" "$kernel_source/kernel.files.sha256"
mkdir "$kernel_source/payload"
cp -R "$payload/." "$kernel_source/payload/"

if [ -n "$fused_residual_argument" ]; then
  augmented_kernel_source=$scratch/fused-residual-kernel-source
  "$repo_root/scripts/augment-luna-kernel-root-plan-with-fused-runtime.sh" \
    "$kernel_source#sha256=$kernel_plan_sha" \
    "$fused_residual#sha256=$fused_residual_sha" \
    "$augmented_kernel_source" >"$scratch/kernel-augment.stdout" \
    2>"$scratch/kernel-augment.stderr" ||
    fail 'reusable fused residual kernel-root augmentation failed'
  [ ! -s "$scratch/kernel-augment.stderr" ] ||
    fail 'reusable fused residual kernel-root augmentation emitted stderr'
  kernel_source=$augmented_kernel_source
  inventory_sha=$(sed -n 's/^kernel_inventory_sha256=//p' \
    "$scratch/kernel-augment.stdout")
  kernel_plan_sha=$(sed -n 's/^kernel_plan_sha256=//p' \
    "$scratch/kernel-augment.stdout")
  bundle_is_lower_sha256 "$inventory_sha" &&
    bundle_is_lower_sha256 "$kernel_plan_sha" ||
    fail 'reusable fused residual kernel-root augmentation emitted invalid digests'
fi

mkdir "$output"
output_created=1
mkdir -p "$output/model-root/runtime" "$output/policy-root/instance" \
  "$output/bin" "$output/evidence" "$output/corpus"
"$repo_root/scripts/assemble-luna-kernel-root.sh" \
  "$kernel_source#sha256=$kernel_plan_sha" "$output/kernel-release" \
  >"$scratch/kernel-assemble.stdout" 2>"$scratch/kernel-assemble.stderr" ||
  fail 'kernel-root assembly failed'
"$repo_root/scripts/verify-luna-kernel-root.sh" "$output/kernel-release" >/dev/null
[ "$(bundle_sha256_file "$output/kernel-release/kernel.files.sha256")" = \
  "$inventory_sha" ] || fail 'assembled kernel inventory mismatch'

cp "$config" "$output/model-root/config.json"
cp "$tokenizer" "$output/model-root/tokenizer.json"
cp "$numeric" "$output/model-root/$numeric_locator"
cp "$worker" "$output/bin/lunaflux-device-worker"
chmod 555 "$output/bin/lunaflux-device-worker"
cp "$corpus" "$output/corpus/reference.json"
cp "$bind_stdout" "$output/evidence/release-bind.v1"
cp "$scratch/kernel-assemble.stdout" "$output/evidence/kernel-assemble.stdout"
cp "$scratch/kernel-assemble.stderr" "$output/evidence/kernel-assemble.stderr"

descriptor=$output/model-root/runtime/descriptor.json
printf '%s\n' "{\"schema_version\":\"$descriptor_schema\",\"model\":{\"family\":\"qwen3\",\"config_locator\":\"config.json\",\"config_sha256\":\"$config_sha\",\"content_sha256\":\"$model_content_sha\",\"numeric_weights_locator\":\"$numeric_locator\",\"numeric_weight_artifact_sha256\":\"$numeric_sha\",\"weight_manifest_sha256\":\"$route_sha\",\"tied_embeddings\":true,\"max_batch_rows\":$max_batch_rows},\"kernels\":{\"manifest_locator\":\"$manifest_relative\",\"manifest_sha256\":\"$manifest_sha\",\"policy\":\"deployment_approved_aot_only\",\"admitted_bootstrap_sha256\":\"$bootstrap_sha\"$sampling_runtime_json$fused_runtime_json},\"execution\":{\"device_ordinal\":0,\"compute_major\":12,\"compute_minor\":0,\"supports_bf16\":true,\"supports_cublas_lt\":false,\"tokens_per_page\":$tokens_per_page,\"total_page_count\":$total_page_count,\"model_generation\":1},\"worker_limits\":{\"max_prefill_rows\":$max_prefill_rows,\"max_decode_rows\":$max_decode_rows,\"max_plan_rows\":$max_plan_rows,\"max_plan_tokens\":$max_plan_tokens,\"max_plan_pages\":$max_plan_pages,\"max_capabilities\":1024,\"max_completion_slots\":$max_completion_slots,\"max_sequence_tokens\":$max_sequence_tokens,\"max_token_id\":151935},\"inference_limits\":{\"max_text_bytes\":65536,\"max_input_tokens\":4096,\"max_new_tokens\":256,\"max_context_tokens\":$max_sequence_tokens,\"max_token_id\":151935,\"max_stop_token_ids\":16,\"max_stop_strings\":16,\"max_stop_string_bytes\":256,\"max_trace_bytes\":128,\"max_cache_scope_bytes\":64,\"max_decoded_delta_bytes\":$max_decoded_delta_bytes,\"max_deadline_millis\":60000,\"max_top_k\":151936,\"max_temperature\":2.0},\"ceilings\":{\"max_model_config_bytes\":1048576,\"max_weight_file_bytes\":3221225472,\"max_weight_arena_bytes\":3221225472,\"max_activation_arena_bytes\":2147483648,\"max_kv_arena_bytes\":17179869184,\"max_execution_manifest_bytes\":1048576,\"max_module_bytes\":4194304,\"max_total_module_bytes\":2147483647}}" >"$descriptor"
descriptor_sha=$(bundle_sha256_file "$descriptor")

policy=$output/policy-root/instance/policy.json
printf '%s\n' "{\"schema_version\":\"$policy_schema\",\"runtime_descriptor_sha256\":\"$descriptor_sha\",\"tokenizer\":{\"locator\":\"tokenizer.json\",\"sha256\":\"$tokenizer_sha\",\"max_file_bytes\":67108864,\"max_json_depth\":64,\"max_input_bytes\":65536,\"max_output_tokens\":65536,\"max_decoded_bytes\":1048576,\"max_vocab_entries\":200000,\"max_merge_rules\":1000000,\"max_token_bytes\":$tokenizer_max_token_bytes,\"max_special_tokens\":64},\"scheduler\":{\"step_token_budget\":$scheduler_step_token_budget,\"max_active_requests\":$scheduler_max_active_requests,\"max_waiting_requests\":$scheduler_max_waiting_requests,\"prefill_chunk_tokens\":$scheduler_prefill_chunk_tokens,\"emergency_decode_page_reserve\":1,\"waiting_age_threshold_steps\":8,\"output_event_capacity\":$scheduler_output_event_capacity},\"cache\":{\"tokens_per_page\":$tokens_per_page,\"total_page_count\":$total_page_count,\"block_table_pages_per_request\":$context_page_table_entries,\"prefix_enabled\":false,\"max_prefix_entries\":0,\"max_prefix_nodes\":0,\"max_prefix_tokens_per_entry\":0,\"max_prefix_pages\":0,\"max_prefix_scope_bytes\":0,\"max_active_references_per_page\":1,\"max_cached_references_per_page\":1,\"layout_version\":1},\"service\":{\"listen_host\":\"127.0.0.1\",\"listen_port\":$listen_port,\"max_request_bytes\":1048576,\"graceful_drain_milliseconds\":30000,\"diagnostic_mode\":\"normal\"}$external_protocol_json$openai_service_json,\"worker_process\":{\"max_frame_bytes\":1048576,\"startup_io_timeout_millis\":600000,\"io_timeout_millis\":600000,\"shutdown_timeout_millis\":600000},\"restart\":{\"initial_backoff_millis\":10,\"maximum_backoff_millis\":1000,\"stable_after_millis\":1000,\"maximum_attempts\":3},\"transport\":{\"read_chunk_bytes\":65536,\"write_chunk_bytes\":65536,\"accept_timeout_millis\":1000,\"input_idle_timeout_millis\":60000,\"write_timeout_millis\":60000,\"reactor_transition_budget\":256},\"preparation\":{\"framed_max_frame_bytes\":1048576,\"lane_count\":$preparation_lane_count,\"step_work_units\":256,\"total_work_units\":10000000,\"storage_int_cells\":$preparation_storage_int_cells,\"storage_byte_cells\":$preparation_storage_byte_cells,\"storage_reference_cells\":200000,\"event_work_units\":256},\"telemetry\":{\"instance_log_capacity\":4096}}" >"$policy"
policy_sha=$(bundle_sha256_file "$policy")

launch=$output/lunaflux.launch.json
printf '%s\n' "{\"schema\":\"lunaflux.launch.v5\",\"runtime_recipe\":\"dense_qwen3_bf16_paged_aot_v12\",\"model_root\":\"$output/model-root\",\"kernel_root\":\"$output/kernel-release/kernel-root\",\"policy_root\":\"$output/policy-root\",\"runtime_descriptor\":{\"locator\":\"runtime/descriptor.json\",\"sha256\":\"$descriptor_sha\"},\"instance_policy\":{\"locator\":\"instance/policy.json\",\"sha256\":\"$policy_sha\"},\"worker_executable\":{\"path\":\"$output/bin/lunaflux-device-worker\",\"sha256\":\"$worker_sha\"},\"luna_approval\":{\"mode\":\"none\"}}" >"$launch"
launch_sha=$(bundle_sha256_file "$launch")

if ! moon run --target native cmd/lunaflux -- validate-release \
  "$output#sha256=$launch_sha" >"$scratch/preflight.stdout" \
  2>"$scratch/preflight.stderr"; then
  sed -n '1,160p' "$scratch/preflight.stdout" >&2
  sed -n '1,160p' "$scratch/preflight.stderr" >&2
  fail 'Qwen release preflight failed'
fi
for exact in \
  'schema=lunaflux-release-preflight.v1' \
  'runtime_recipe=dense_qwen3_bf16_paged_aot_v12' \
  "launch_sha256=$launch_sha" \
  "runtime_descriptor_sha256=$descriptor_sha" \
  "instance_policy_sha256=$policy_sha" \
  "tokenizer_sha256=$tokenizer_sha" \
  "worker_executable_sha256=$worker_sha" \
  "model_content_sha256=$model_content_sha" \
  "model_plan_sha256=$model_plan_sha" \
  "bootstrap_sha256=$bootstrap_sha" \
  'device_opened=0' 'compiler_jit_authority=0'; do
  grep -Fxq "$exact" "$scratch/preflight.stdout" ||
    fail "release preflight lost: $exact"
done
cp "$scratch/preflight.stdout" "$output/evidence/release-preflight.v1"
cp "$scratch/preflight.stderr" "$output/evidence/release-preflight.stderr"

inventory_out=$output/bundle.files.sha256
find "$output" -type f ! -path "$inventory_out" -print |
  sed "s#^$output/##" | LC_ALL=C sort |
  while IFS= read -r relative; do
    printf '%s  %s\n' "$(bundle_sha256_file "$output/$relative")" "$relative"
  done >"$inventory_out"
find "$output" -type f ! -path "$output/bin/lunaflux-device-worker" -exec chmod 444 {} \;
find "$output" -type d -exec chmod 555 {} \;
chmod 755 "$output"

complete=1
trap - EXIT HUP INT TERM
rm -rf -- "$scratch"
printf '%s\n' 'schema=lunaflux-qwen3-v12-launch-materialization.v1'
printf 'deployment_root=%s\n' "$output"
printf 'launch_sha256=%s\n' "$launch_sha"
printf 'model_plan_sha256=%s\n' "$model_plan_sha"
printf 'bootstrap_sha256=%s\n' "$bootstrap_sha"
printf 'bootstrap_source_sha256=%s\n' "$bootstrap_source_sha"
printf 'reference_corpus_sha256=%s\n' "$corpus_sha"
printf 'listen_addr=127.0.0.1:%s\n' "$listen_port"
printf 'sampling_runtime=%s\n' "$sampling_runtime"
printf 'materialization_profile=%s\n' "$materialization_profile"
printf 'external_protocol=%s\n' "$(if [ "$materialization_profile" = openai-responses-v1 ]; then printf '%s' openai-responses-v1; else printf '%s' native-framed-v1; fi)"
printf 'max_batch_rows=%s\n' "$max_batch_rows"
printf 'scheduler_max_active_requests=%s\n' "$scheduler_max_active_requests"
printf 'scheduler_max_waiting_requests=%s\n' "$scheduler_max_waiting_requests"
