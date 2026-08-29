#!/bin/sh

set -eu
LC_ALL=C
TZ=UTC
export LC_ALL TZ
umask 077

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
. "$repo_root/scripts/release-bundle-common.sh"

fail() {
  printf '%s\n' "LunaFlux approved BF16 launch materialization rejected: $1" >&2
  exit 1
}

usage() {
  printf '%s\n' \
    'usage: materialize-approved-tiny-bf16-launch.sh ABSOLUTE_MODEL_ROOT ABSOLUTE_COMPILED_SET_ROOT ABSOLUTE_TOOLCHAIN_MANIFEST#sha256=HEX COMPILER_MAJOR COMPILER_MINOR COMPILER_PATCH ABSOLUTE_WORKER#sha256=HEX ABSOLUTE_NEW_OUTPUT [prefix-reuse-v1|openai-responses-v1|host-sampling-referee-v1|fused-v2-device-greedy-v1 RUNTIME#sha QKV#sha READONLY#sha RESIDUAL#sha IDENTITIES#sha CAMPAIGN_DIR OUTER_SHA POLICY_SHA]' >&2
  exit 2
}

case "$#" in 8|9|17) ;; *) usage ;; esac
model_root=$1
compiled_root=$2
toolchain_argument=$3
compiler_major=$4
compiler_minor=$5
compiler_patch=$6
worker_argument=$7
output=$8
materialization_profile=baseline-v1
descriptor_max_input_tokens=8
tokenizer_max_output_tokens=8
prefix_enabled=false
max_prefix_entries=0
max_prefix_nodes=0
max_prefix_tokens_per_entry=0
max_prefix_pages=0
max_prefix_scope_bytes=0
policy_schema=lunaflux.instance-policy.v1
external_protocol_json=
openai_service_json=
sampling_runtime_json=',"sampling_runtime":"embedded_cuda_greedy_v1"'
fused_runtime_json=
descriptor_schema=lunaflux.runtime.v4
fused_runtime_argument=
if [ "$#" -eq 9 ] || [ "$#" -eq 17 ]; then
  case "$9" in
    prefix-reuse-v1)
      [ "$#" -eq 9 ] || usage
      materialization_profile=prefix-reuse-v1
      descriptor_max_input_tokens=9
      tokenizer_max_output_tokens=9
      prefix_enabled=true
      max_prefix_entries=1
      max_prefix_nodes=8
      max_prefix_tokens_per_entry=8
      max_prefix_pages=1
      max_prefix_scope_bytes=32
      ;;
    openai-responses-v1)
      [ "$#" -eq 9 ] || usage
      materialization_profile=openai-responses-v1
      policy_schema=lunaflux.instance-policy.v3
      external_protocol_json=',"external_protocol":{"mode":"openai_responses_v1","transport_security":"loopback_plaintext","control_authentication":"deployment_bearer","invocation_path":"/v1/responses","health_path":"/healthz","readiness_path":"/readyz","drain_path":"/v1/drain","drain_method":"POST"}'
      openai_service_json=',"openai_service":{"maximum_credential_bytes":128,"max_head_bytes":4096,"max_body_bytes":4096,"http_step_work_units":64,"response_step_work_units":64,"inbound_step_work_units":64,"outbound_step_work_units":64,"max_messages":2,"system_prefix":"","system_suffix":"","user_prefix":"","user_suffix":"","assistant_prefix":"","assistant_suffix":"","assistant_cue":"","max_rendered_prompt_bytes":1024,"model_alias":"tiny-bf16","response_id_prefix":"resp_lunaflux_","cache_scope_ascii":"openai-v3","max_new_tokens":2,"context_ceiling":8,"sampling_seed":7,"deadline_milliseconds":60000}'
      ;;
    host-sampling-referee-v1)
      [ "$#" -eq 9 ] || usage
      materialization_profile=host-sampling-referee-v1
      sampling_runtime_json=
      descriptor_schema=lunaflux.runtime.v3
      ;;
    fused-v2-device-greedy-v1)
      [ "$#" -eq 17 ] || usage
      materialization_profile=fused-v2-device-greedy-v1
      fused_runtime_argument=${10}
      qkv_argument=${11}
      readonly_argument=${12}
      residual_argument=${13}
      identities_argument=${14}
      fused_campaign_root=${15}
      fused_campaign_outer_sha=${16}
      fused_policy_sha=${17}
      ;;
    *) usage ;;
  esac
fi

case "$materialization_profile" in
  baseline-v1|prefix-reuse-v1|openai-responses-v1|host-sampling-referee-v1)
    [ "$#" -le 9 ] || usage
    ;;
esac

case "$toolchain_argument" in /*#sha256=*) ;; *) usage ;; esac
toolchain_manifest=${toolchain_argument%#sha256=*}
toolchain_sha=${toolchain_argument##*#sha256=}
case "$worker_argument" in /*#sha256=*) ;; *) usage ;; esac
worker_source=${worker_argument%#sha256=*}
worker_sha=${worker_argument##*#sha256=}

if [ -n "$fused_runtime_argument" ]; then
  for pinned in "$fused_runtime_argument" "$qkv_argument" \
    "$readonly_argument" "$residual_argument" "$identities_argument"; do
    case "$pinned" in /*#sha256=*) ;; *) usage ;; esac
  done
  fused_runtime_source=${fused_runtime_argument%#sha256=*}
  fused_runtime_sha=${fused_runtime_argument##*#sha256=}
  qkv_source=${qkv_argument%#sha256=*}
  qkv_sha=${qkv_argument##*#sha256=}
  readonly_source=${readonly_argument%#sha256=*}
  readonly_sha=${readonly_argument##*#sha256=}
  residual_source=${residual_argument%#sha256=*}
  residual_sha=${residual_argument##*#sha256=}
  identities_source=${identities_argument%#sha256=*}
  identities_sha=${identities_argument##*#sha256=}
  fused_runtime_json=',"fused_runtime":{"locator":"fused-production.runtime.v1","sha256":"'"$fused_runtime_sha"'"}'
fi

bundle_require_canonical_absolute_directory "$model_root"
bundle_require_canonical_absolute_directory "$compiled_root"
bundle_require_canonical_absolute_file "$toolchain_manifest"
bundle_require_canonical_absolute_file "$worker_source"
bundle_require_digest "$toolchain_manifest" "$toolchain_sha" 'toolchain manifest'
bundle_require_digest "$worker_source" "$worker_sha" 'worker executable'
[ -x "$worker_source" ] || fail 'worker executable is not executable'
for component in "$compiler_major" "$compiler_minor" "$compiler_patch"; do
  case "$component" in ''|*[!0-9]*) fail 'compiler version is invalid' ;; esac
  [ "${#component}" -le 6 ] || fail 'compiler version is out of bounds'
done
[ "$compiler_major" -gt 0 ] || fail 'compiler major version must be positive'

if [ -n "$fused_runtime_argument" ]; then
  bundle_require_canonical_absolute_directory "$fused_campaign_root"
  bundle_require_canonical_absolute_file "$fused_runtime_source"
  for input in "$qkv_source" "$readonly_source" \
    "$residual_source" "$identities_source"; do
    bundle_require_canonical_absolute_file "$input"
    case "$input" in "$fused_campaign_root"/*) ;; *)
      fail 'fused join input is outside the sealed lower campaign'
      ;;
    esac
  done
  bundle_require_digest "$fused_runtime_source" "$fused_runtime_sha" 'fused runtime'
  bundle_require_digest "$qkv_source" "$qkv_sha" 'fused qkv module'
  bundle_require_digest "$readonly_source" "$readonly_sha" 'readonly attention module'
  bundle_require_digest "$residual_source" "$residual_sha" 'fused residual module'
  bundle_require_digest "$identities_source" "$identities_sha" 'fused identities'
  bundle_is_lower_sha256 "$fused_campaign_outer_sha" ||
    fail 'fused lower-campaign outer seal pin is invalid'
  bundle_is_lower_sha256 "$fused_policy_sha" ||
    fail 'fused lower-campaign policy pin is invalid'
  "$repo_root/scripts/verify-fused-production-v2-physical-campaign.sh" \
    "$fused_campaign_root" "$fused_campaign_outer_sha" "$fused_policy_sha" \
    >/dev/null || fail 'sealed lower fused production campaign rejected'
fi

approved_config_sha=9197475bfcc987a4f9361dbc22b33397b101372c137c228b6a6fd7e4adf21622
approved_model_sha=852db1b39acb2336abc997440c6f6d6e4ab640f91e5e2aa9e2488d5794159d30
approved_tokenizer_sha=fbcdbe15960e43ef351662e7b77a319ceb294b3c5dc2569c23b729fb87e13d7b
for input in config.json model.safetensors upstream_tokenizer.json; do
  bundle_require_canonical_absolute_file "$model_root/$input"
done
bundle_require_digest "$model_root/config.json" "$approved_config_sha" 'model config'
bundle_require_digest "$model_root/model.safetensors" "$approved_model_sha" 'model weights'
bundle_require_digest "$model_root/upstream_tokenizer.json" "$approved_tokenizer_sha" 'tokenizer'
"$repo_root/scripts/verify-luna-bf16-kernel-set.sh" "$compiled_root" >/dev/null

case "$output" in /*) ;; *) fail 'output path must be absolute' ;; esac
case "$output" in /|*//*|*/./*|*/../*|*/.|*/..|*[!A-Za-z0-9._/-]*)
  fail 'output path is not a safe canonical absolute path'
  ;;
esac
[ ! -e "$output" ] && [ ! -L "$output" ] || fail 'refusing to overwrite an existing output'
output_parent=$(CDPATH= cd -- "$(dirname -- "$output")" && pwd -P)
[ "$output_parent/$(basename -- "$output")" = "$output" ] ||
  fail 'output parent is not canonical'

scratch=$(mktemp -d /tmp/lunaflux-approved-bf16-launch.XXXXXX) ||
  fail 'could not create private scratch'
mkdir "$output" || fail 'could not claim new output'
claim=$output/.materialization-claim
printf '%s\n' 'lunaflux-approved-tiny-bf16-launch-claim-v1' >"$claim"
complete=0
cleanup() {
  rm -rf -- "$scratch"
  if [ "$complete" -ne 1 ] && [ -f "$claim" ] &&
    [ "$(sed -n '1p' "$claim")" = lunaflux-approved-tiny-bf16-launch-claim-v1 ]; then
    chmod -R u+w "$output" 2>/dev/null || true
    rm -rf -- "$output"
  fi
}
trap cleanup EXIT HUP INT TERM

if [ -n "$fused_runtime_argument" ]; then
  inspection=$scratch/fused-runtime-inspection.v1
  if ! moon run --target native tests/approved_model_spawned_physical -- \
    inspect-fused-production-runtime "$fused_runtime_source" "$fused_runtime_sha" \
    >"$inspection" 2>"$scratch/fused-runtime-inspection.stderr"; then
    sed -n '1,120p' "$scratch/fused-runtime-inspection.stderr" >&2
    fail 'aggregate fused production runtime did not pass normal admission'
  fi
  [ ! -s "$scratch/fused-runtime-inspection.stderr" ] ||
    fail 'aggregate fused production runtime inspection emitted stderr'
  inspection_value() {
    iv=$(sed -n "s/^$1=//p" "$inspection")
    [ -n "$iv" ] && [ "$(grep -c "^$1=" "$inspection")" -eq 1 ] ||
      fail "fused runtime inspection field is invalid: $1"
    printf '%s\n' "$iv"
  }
  identity_value() {
    iv=$(sed -n "s/^$1=//p" "$identities_source")
    [ -n "$iv" ] && [ "$(grep -c "^$1=" "$identities_source")" -eq 1 ] ||
      fail "lower fused identity field is invalid: $1"
    printf '%s\n' "$iv"
  }
  [ "$(inspection_value qkv_module_sha256)" = "$qkv_sha" ] &&
    [ "$(inspection_value readonly_module_sha256)" = "$readonly_sha" ] &&
    [ "$(inspection_value residual_module_sha256)" = "$residual_sha" ] ||
    fail 'aggregate runtime substituted a lower-campaign module'
  for family in qkv readonly residual; do
    [ "$(inspection_value "${family}_source_sha256")" = \
      "$(identity_value "production_${family}_source_sha256")" ] ||
      fail "aggregate runtime substituted the $family source identity"
    [ "$(inspection_value "${family}_symbol")" = \
      "$(identity_value "production_${family}_symbol")" ] ||
      fail "aggregate runtime substituted the $family symbol"
  done
  [ "$(inspection_value normal_descriptor_admission)" = pass ] &&
    [ "$(inspection_value qualification_only)" = true ] &&
    [ "$(inspection_value promotion_authority)" = absent ] ||
    fail 'aggregate runtime inspection gained promotion authority'
fi

kernel_source=$output/evidence/kernel-source
mkdir -p "$output/evidence" "$output/model-root/runtime" \
  "$output/policy-root/instance" "$output/bin"

if ! moon run --target native cmd/lunaflux_bf16_release_bind -- \
  "$model_root" "$compiled_root" "$toolchain_manifest" "$toolchain_sha" \
  "$compiler_major" "$compiler_minor" "$compiler_patch" "$kernel_source" \
  >"$scratch/bind.stdout" 2>"$scratch/bind.stderr"; then
  sed -n '1,160p' "$scratch/bind.stdout" >&2
  sed -n '1,160p' "$scratch/bind.stderr" >&2
  fail 'typed compiled-set binding failed'
fi
[ "$(wc -l <"$scratch/bind.stdout" | tr -d ' ')" -eq 17 ] ||
  fail 'typed binder evidence has the wrong field count'
bind_value() {
  bind_line=$(sed -n "$1p" "$scratch/bind.stdout")
  case "$bind_line" in "$2="*) printf '%s\n' "${bind_line#*=}" ;; *)
    fail "typed binder evidence field $2 is not canonical"
    ;;
  esac
}
[ "$(bind_value 1 schema)" = lunaflux-approved-bf16-release-bind.v1 ] ||
  fail 'typed binder schema is unsupported'
[ "$(bind_value 2 model_content_sha256)" = "$approved_model_sha" ] ||
  fail 'typed binder model identity mismatch'
model_plan_sha=$(bind_value 3 model_plan_sha256)
[ "$(bind_value 4 target)" = sm_120 ] &&
  [ "$(bind_value 5 tokens_per_page)" = 8 ] &&
  [ "$(bind_value 6 total_page_count)" = 32 ] &&
  [ "$(bind_value 7 max_sequence_tokens)" = 256 ] &&
  [ "$(bind_value 8 max_page_table_entries)" = 32 ] ||
  fail 'typed binder runtime geometry mismatch'
manifest_relative=$(bind_value 9 kernel_manifest_relative)
manifest_sha=$(bind_value 10 kernel_manifest_sha256)
bootstrap_sha=$(bind_value 11 admitted_bootstrap_sha256)
kernel_inventory_sha=$(bind_value 12 kernel_inventory_sha256)
kernel_plan_sha=$(bind_value 13 kernel_root_plan_sha256)
for digest in "$model_plan_sha" "$manifest_sha" "$bootstrap_sha" \
  "$kernel_inventory_sha" "$kernel_plan_sha"; do
  bundle_is_lower_sha256 "$digest" || fail 'typed binder emitted an invalid digest'
done
[ "$manifest_relative" = lunaflux.execution.json ] ||
  fail 'typed binder emitted an invalid manifest locator'
[ "$(bind_value 14 compiler_invoked)" = 0 ] &&
  [ "$(bind_value 15 device_opened)" = 0 ] &&
  [ "$(bind_value 16 runtime_authority)" = 0 ] ||
  fail 'typed binder crossed its offline authority boundary'
[ "$(bind_value 17 authenticated_embedded_greedy_sampling)" = 1 ] ||
  fail 'typed release did not authenticate the embedded greedy reducer'

if [ -n "$fused_runtime_argument" ]; then
  augmented_kernel_source=$scratch/fused-kernel-source
  "$repo_root/scripts/augment-luna-kernel-root-plan-with-fused-runtime.sh" \
    "$kernel_source#sha256=$kernel_plan_sha" \
    "$fused_runtime_source#sha256=$fused_runtime_sha" \
    "$augmented_kernel_source" >"$scratch/kernel-augment.stdout" \
    2>"$scratch/kernel-augment.stderr" ||
    fail 'fused runtime kernel-root augmentation failed'
  [ ! -s "$scratch/kernel-augment.stderr" ] ||
    fail 'fused runtime kernel-root augmentation emitted stderr'
  kernel_source=$augmented_kernel_source
  kernel_inventory_sha=$(sed -n 's/^kernel_inventory_sha256=//p' \
    "$scratch/kernel-augment.stdout")
  kernel_plan_sha=$(sed -n 's/^kernel_plan_sha256=//p' \
    "$scratch/kernel-augment.stdout")
  bundle_is_lower_sha256 "$kernel_inventory_sha" &&
    bundle_is_lower_sha256 "$kernel_plan_sha" ||
    fail 'fused runtime kernel-root augmentation emitted invalid digests'
fi

"$repo_root/scripts/assemble-luna-kernel-root.sh" \
  "$kernel_source#sha256=$kernel_plan_sha" "$output/kernel-release" \
  >"$scratch/kernel-assemble.stdout" 2>"$scratch/kernel-assemble.stderr" ||
  fail 'kernel-root assembly failed'
"$repo_root/scripts/verify-luna-kernel-root.sh" "$output/kernel-release" >/dev/null
[ "$(bundle_sha256_file "$output/kernel-release/kernel.files.sha256")" = \
  "$kernel_inventory_sha" ] || fail 'assembled kernel inventory mismatch'
[ "$(bundle_sha256_file "$output/kernel-release/kernel-root/$manifest_relative")" = \
  "$manifest_sha" ] || fail 'assembled execution manifest mismatch'

cp "$model_root/config.json" "$output/model-root/config.json"
cp "$model_root/model.safetensors" "$output/model-root/model.safetensors"
cp "$model_root/upstream_tokenizer.json" "$output/model-root/upstream_tokenizer.json"
cp "$worker_source" "$output/bin/lunaflux-device-worker"
chmod 555 "$output/bin/lunaflux-device-worker"

descriptor=$output/model-root/runtime/descriptor.json
printf '%s\n' "{\"schema_version\":\"$descriptor_schema\",\"model\":{\"config_locator\":\"config.json\",\"config_sha256\":\"$approved_config_sha\",\"weights_locator\":\"model.safetensors\",\"content_sha256\":\"$approved_model_sha\",\"max_batch_rows\":1},\"kernels\":{\"manifest_locator\":\"$manifest_relative\",\"manifest_sha256\":\"$manifest_sha\",\"policy\":\"deployment_approved_aot_only\",\"admitted_bootstrap_sha256\":\"$bootstrap_sha\"$fused_runtime_json$sampling_runtime_json},\"execution\":{\"device_ordinal\":0,\"compute_major\":12,\"compute_minor\":0,\"supports_bf16\":true,\"supports_cublas_lt\":true,\"tokens_per_page\":8,\"total_page_count\":32,\"model_generation\":1},\"worker_limits\":{\"max_prefill_rows\":1,\"max_decode_rows\":1,\"max_plan_rows\":1,\"max_plan_tokens\":1,\"max_plan_pages\":32,\"max_capabilities\":64,\"max_completion_slots\":1,\"max_sequence_tokens\":256,\"max_token_id\":2999},\"inference_limits\":{\"max_text_bytes\":1024,\"max_input_tokens\":$descriptor_max_input_tokens,\"max_new_tokens\":8,\"max_context_tokens\":256,\"max_token_id\":2999,\"max_stop_token_ids\":4,\"max_stop_strings\":4,\"max_stop_string_bytes\":32,\"max_trace_bytes\":32,\"max_cache_scope_bytes\":32,\"max_decoded_delta_bytes\":512,\"max_deadline_millis\":60000,\"max_top_k\":3000,\"max_temperature\":2.0},\"ceilings\":{\"max_model_config_bytes\":65536,\"max_weight_file_bytes\":210712,\"max_weight_arena_bytes\":1048576,\"max_activation_arena_bytes\":33554432,\"max_kv_arena_bytes\":1048576,\"max_execution_manifest_bytes\":1048576,\"max_module_bytes\":4194304,\"max_total_module_bytes\":67108864,\"max_graph_capture_bytes\":33554432}}" >"$descriptor"
descriptor_sha=$(bundle_sha256_file "$descriptor")

policy=$output/policy-root/instance/policy.json
printf '%s\n' "{\"schema_version\":\"$policy_schema\",\"runtime_descriptor_sha256\":\"$descriptor_sha\",\"tokenizer\":{\"locator\":\"upstream_tokenizer.json\",\"sha256\":\"$approved_tokenizer_sha\",\"max_file_bytes\":65536,\"max_json_depth\":16,\"max_input_bytes\":1024,\"max_output_tokens\":$tokenizer_max_output_tokens,\"max_decoded_bytes\":512,\"max_vocab_entries\":3000,\"max_merge_rules\":1,\"max_token_bytes\":256,\"max_special_tokens\":4},\"scheduler\":{\"step_token_budget\":1,\"max_active_requests\":1,\"max_waiting_requests\":1,\"prefill_chunk_tokens\":1,\"emergency_decode_page_reserve\":1,\"waiting_age_threshold_steps\":8,\"output_event_capacity\":8},\"cache\":{\"tokens_per_page\":8,\"total_page_count\":32,\"block_table_pages_per_request\":32,\"prefix_enabled\":$prefix_enabled,\"max_prefix_entries\":$max_prefix_entries,\"max_prefix_nodes\":$max_prefix_nodes,\"max_prefix_tokens_per_entry\":$max_prefix_tokens_per_entry,\"max_prefix_pages\":$max_prefix_pages,\"max_prefix_scope_bytes\":$max_prefix_scope_bytes,\"max_active_references_per_page\":1,\"max_cached_references_per_page\":1,\"layout_version\":1},\"service\":{\"listen_host\":\"127.0.0.1\",\"listen_port\":8080,\"max_request_bytes\":1048576,\"graceful_drain_milliseconds\":1000,\"diagnostic_mode\":\"normal\"}$external_protocol_json$openai_service_json,\"worker_process\":{\"max_frame_bytes\":1048576,\"startup_io_timeout_millis\":60000,\"io_timeout_millis\":60000,\"shutdown_timeout_millis\":60000},\"restart\":{\"initial_backoff_millis\":1,\"maximum_backoff_millis\":1000,\"stable_after_millis\":1000,\"maximum_attempts\":3},\"transport\":{\"read_chunk_bytes\":4096,\"write_chunk_bytes\":4096,\"accept_timeout_millis\":1000,\"input_idle_timeout_millis\":60000,\"write_timeout_millis\":60000,\"reactor_transition_budget\":64},\"preparation\":{\"framed_max_frame_bytes\":1048576,\"lane_count\":2,\"step_work_units\":64,\"total_work_units\":100000,\"storage_int_cells\":1000000,\"storage_byte_cells\":4194304,\"storage_reference_cells\":100000,\"event_work_units\":64},\"telemetry\":{\"instance_log_capacity\":1024}}" >"$policy"
policy_sha=$(bundle_sha256_file "$policy")

launch=$output/lunaflux.launch.json
printf '%s\n' "{\"schema\":\"lunaflux.launch.v2\",\"runtime_recipe\":\"dense_llama_paged_aot_v5\",\"model_root\":\"$output/model-root\",\"kernel_root\":\"$output/kernel-release/kernel-root\",\"policy_root\":\"$output/policy-root\",\"runtime_descriptor\":{\"locator\":\"runtime/descriptor.json\",\"sha256\":\"$descriptor_sha\"},\"instance_policy\":{\"locator\":\"instance/policy.json\",\"sha256\":\"$policy_sha\"},\"worker_executable\":{\"path\":\"$output/bin/lunaflux-device-worker\",\"sha256\":\"$worker_sha\"},\"luna_approval\":{\"mode\":\"none\"}}" >"$launch"
launch_sha=$(bundle_sha256_file "$launch")

cp "$scratch/bind.stdout" "$output/evidence/release-bind.v1"
cp "$scratch/bind.stderr" "$output/evidence/release-bind.stderr"
cp "$scratch/kernel-assemble.stdout" "$output/evidence/kernel-assemble.stdout"
cp "$scratch/kernel-assemble.stderr" "$output/evidence/kernel-assemble.stderr"
if [ -n "$fused_runtime_argument" ]; then
  cp "$inspection" "$output/evidence/fused-runtime-inspection.v1"
  cp "$scratch/kernel-augment.stdout" "$output/evidence/kernel-augment.v1"
  cp "$identities_source" "$output/evidence/fused-production-identities.v1"
fi

if ! moon run --target native cmd/lunaflux -- validate-release \
  "$output#sha256=$launch_sha" >"$scratch/preflight.stdout" 2>"$scratch/preflight.stderr"; then
  sed -n '1,160p' "$scratch/preflight.stdout" >&2
  sed -n '1,160p' "$scratch/preflight.stderr" >&2
  fail 'production semantic release preflight failed'
fi
for exact in \
  'schema=lunaflux-release-preflight.v1' \
  'runtime_recipe=dense_llama_paged_aot_v5' \
  "launch_sha256=$launch_sha" \
  "runtime_descriptor_sha256=$descriptor_sha" \
  "instance_policy_sha256=$policy_sha" \
  "tokenizer_sha256=$approved_tokenizer_sha" \
  "worker_executable_sha256=$worker_sha" \
  "model_content_sha256=$approved_model_sha" \
  "model_plan_sha256=$model_plan_sha" \
  "bootstrap_sha256=$bootstrap_sha" \
  'device_ordinal=0' 'compute_major=12' 'compute_minor=0' \
  'semantic_join=1' 'filesystem_authority_closed=1' \
  'device_opened=0' 'compiler_jit_authority=0'; do
  grep -Fxq "$exact" "$scratch/preflight.stdout" ||
    fail "semantic preflight evidence lost: $exact"
done
cp "$scratch/preflight.stdout" "$output/evidence/release-preflight.v1"
cp "$scratch/preflight.stderr" "$output/evidence/release-preflight.stderr"

bundle_inventory=$output/bundle.files.sha256
find "$output" -type f ! -path "$claim" ! -path "$bundle_inventory" -print |
  sed "s#^$output/##" | LC_ALL=C sort |
  while IFS= read -r relative; do
    printf '%s  %s\n' "$(bundle_sha256_file "$output/$relative")" "$relative"
  done >"$bundle_inventory"
chmod 444 "$bundle_inventory"
find "$output" -type f ! -path "$output/bin/lunaflux-device-worker" -exec chmod 444 {} \;
find "$output" -type d -exec chmod 555 {} \;
chmod 755 "$output"
rm -f -- "$claim"

complete=1
trap - EXIT HUP INT TERM
rm -rf -- "$scratch"
printf '%s\n' 'LunaFlux approved tiny BF16 launch materialized and semantically preflighted.'
printf 'launch_root=%s\n' "$output"
printf 'launch_sha256=%s\n' "$launch_sha"
printf 'worker_sha256=%s\n' "$worker_sha"
printf 'admitted_bootstrap_sha256=%s\n' "$bootstrap_sha"
printf 'model_plan_sha256=%s\n' "$model_plan_sha"
printf 'materialization_profile=%s\n' "$materialization_profile"
printf '%s\n' 'compiler_invoked_by_binder=0'
printf '%s\n' 'device_opened_by_materializer=0'
