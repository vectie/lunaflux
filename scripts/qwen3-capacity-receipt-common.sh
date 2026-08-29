#!/bin/sh

# Shared offline admission for the Qwen3 c32 benchmark capacity receipt.

qcap_sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | sed 's/[[:space:]].*$//'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | sed 's/[[:space:]].*$//'
  else
    qcap_fail 'no SHA-256 utility is available'
  fi
}

qcap_is_sha256() {
  case "$1" in
    *[!0-9a-f]*|'') return 1 ;;
  esac
  [ "${#1}" -eq 64 ]
}

qcap_require_pinned_file() {
  qcap_argument=$1
  qcap_label=$2
  case "$qcap_argument" in /*'#sha256='*) ;; *)
    qcap_fail "$qcap_label identity is malformed"
    ;;
  esac
  qcap_pinned_path=${qcap_argument%#sha256=*}
  qcap_pinned_sha=${qcap_argument##*#sha256=}
  qcap_is_sha256 "$qcap_pinned_sha" ||
    qcap_fail "$qcap_label digest is malformed"
  [ -f "$qcap_pinned_path" ] && [ ! -L "$qcap_pinned_path" ] ||
    qcap_fail "$qcap_label is not a regular file"
  qcap_canonical=$(CDPATH= cd -- "$(dirname -- "$qcap_pinned_path")" && pwd -P)/$(basename -- "$qcap_pinned_path")
  [ "$qcap_canonical" = "$qcap_pinned_path" ] ||
    qcap_fail "$qcap_label path is not canonical"
  [ "$(qcap_sha256_file "$qcap_pinned_path")" = "$qcap_pinned_sha" ] ||
    qcap_fail "$qcap_label digest differs"
}

qcap_bind_value() {
  qcap_field=$1
  qcap_field_value=$(sed -n "s/^$qcap_field=//p" "$qcap_bind_stdout")
  [ -n "$qcap_field_value" ] &&
    [ "$(grep -c "^$qcap_field=" "$qcap_bind_stdout")" -eq 1 ] ||
    qcap_fail "release-bind field is absent or duplicated: $qcap_field"
  printf '%s\n' "$qcap_field_value"
}

qcap_require_positive() {
  case "$2" in ''|*[!0-9]*|0) qcap_fail "$1 is not a positive integer" ;; esac
}

qcap_admit_inputs() {
  qcap_require_pinned_file "$1" 'Qwen release-bind stdout'
  qcap_bind_stdout=$qcap_pinned_path
  qcap_bind_stdout_sha=$qcap_pinned_sha

  qcap_deployment_argument=$2
  case "$qcap_deployment_argument" in /*'#sha256='*) ;; *)
    qcap_fail 'Qwen deployment identity is malformed'
    ;;
  esac
  qcap_deployment_root=${qcap_deployment_argument%#sha256=*}
  qcap_launch_sha=${qcap_deployment_argument##*#sha256=}
  qcap_is_sha256 "$qcap_launch_sha" ||
    qcap_fail 'Qwen deployment launch digest is malformed'
  [ -d "$qcap_deployment_root" ] && [ ! -L "$qcap_deployment_root" ] &&
    [ "$(CDPATH= cd -- "$qcap_deployment_root" && pwd -P)" = "$qcap_deployment_root" ] ||
    qcap_fail 'Qwen deployment root is not canonical'
  qcap_launch=$qcap_deployment_root/lunaflux.launch.json
  [ -f "$qcap_launch" ] && [ ! -L "$qcap_launch" ] &&
    [ "$(qcap_sha256_file "$qcap_launch")" = "$qcap_launch_sha" ] ||
    qcap_fail 'Qwen deployment launch identity differs'
  qcap_deployed_bind=$qcap_deployment_root/evidence/release-bind.v1
  [ -f "$qcap_deployed_bind" ] && [ ! -L "$qcap_deployed_bind" ] &&
    [ "$(qcap_sha256_file "$qcap_deployed_bind")" = "$qcap_bind_stdout_sha" ] ||
    qcap_fail 'Qwen deployment does not contain the exact release binding'

  qcap_require_pinned_file "$3" 'LunaFlux runtime executable'
  qcap_runtime_path=$qcap_pinned_path
  qcap_runtime_sha=$qcap_pinned_sha
  [ -x "$qcap_runtime_path" ] || qcap_fail 'LunaFlux runtime is not executable'
  qcap_require_pinned_file "$4" 'Qwen token-ID bridge executable'
  qcap_bridge_path=$qcap_pinned_path
  qcap_bridge_sha=$qcap_pinned_sha
  [ -x "$qcap_bridge_path" ] || qcap_fail 'Qwen token-ID bridge is not executable'

  [ "$(qcap_bind_value schema)" = lunaflux-qwen3-bf16-release-bind.v1 ] &&
    [ "$(qcap_bind_value recipe)" = dense_qwen3_bf16_paged_aot_v12 ] &&
    [ "$(qcap_bind_value target)" = sm_120 ] ||
    qcap_fail 'release binding is not the Qwen3 BF16 v12 sm120 route'
  qcap_model_content_sha=$(qcap_bind_value model_content_sha256)
  qcap_model_plan_sha=$(qcap_bind_value model_plan_sha256)
  qcap_route_sha=$(qcap_bind_value weight_route_manifest_sha256)
  qcap_manifest_sha=$(qcap_bind_value kernel_manifest_sha256)
  qcap_bootstrap_sha=$(qcap_bind_value admitted_bootstrap_sha256)
  for qcap_digest in "$qcap_model_content_sha" "$qcap_model_plan_sha" \
    "$qcap_route_sha" "$qcap_manifest_sha" "$qcap_bootstrap_sha"; do
    qcap_is_sha256 "$qcap_digest" ||
      qcap_fail 'release binding contains a malformed authority digest'
  done
  qcap_max_batch_rows=$(qcap_bind_value max_batch_rows)
  qcap_max_query_rows=$(qcap_bind_value max_query_rows)
  qcap_max_query_tokens=$(qcap_bind_value max_query_tokens)
  qcap_require_positive max_batch_rows "$qcap_max_batch_rows"
  qcap_require_positive max_query_rows "$qcap_max_query_rows"
  qcap_require_positive max_query_tokens "$qcap_max_query_tokens"
  [ "$qcap_max_batch_rows" -eq 32 ] &&
    [ "$qcap_max_query_rows" -eq 32 ] &&
    [ "$qcap_max_query_tokens" -ge 32 ] ||
    qcap_fail 'release binding is not authenticated c32 geometry'
  [ "$(qcap_bind_value compiler_invoked)" = 0 ] &&
    [ "$(qcap_bind_value device_opened)" = 0 ] &&
    [ "$(qcap_bind_value runtime_authority)" = 0 ] ||
    qcap_fail 'release binder crossed its offline boundary'

  qcap_descriptor=$qcap_deployment_root/model-root/runtime/descriptor.json
  qcap_policy=$qcap_deployment_root/policy-root/instance/policy.json
  for qcap_file in "$qcap_descriptor" "$qcap_policy"; do
    [ -f "$qcap_file" ] && [ ! -L "$qcap_file" ] ||
      qcap_fail 'Qwen deployment runtime configuration is absent'
  done
  grep -F '"schema_version":"lunaflux.runtime.qwen3_bf16.v1"' "$qcap_descriptor" >/dev/null &&
    grep -E '"max_batch_rows":32([,}])' "$qcap_descriptor" >/dev/null ||
    qcap_fail 'Qwen runtime descriptor is not c32'
  grep -F '"schema_version":"lunaflux.instance-policy.v1"' "$qcap_policy" >/dev/null &&
    grep -E '"max_active_requests":32([,}])' "$qcap_policy" >/dev/null &&
    grep -E '"max_waiting_requests":32([,}])' "$qcap_policy" >/dev/null ||
    qcap_fail 'Qwen native-framed policy is not c32'
  if grep -F '"external_protocol"' "$qcap_policy" >/dev/null; then
    qcap_fail 'Qwen c32 benchmark deployment is not native-framed'
  fi
}

qcap_write_authentication() {
  qcap_authentication_output=$1
  {
    printf '%s\n' 'schema=lunaflux.qwen3-authenticated-capacity-authentication.v1'
    printf 'release_bind_stdout_sha256=%s\n' "$qcap_bind_stdout_sha"
    printf 'launch_sha256=%s\n' "$qcap_launch_sha"
    printf 'model_content_sha256=%s\n' "$qcap_model_content_sha"
    printf 'model_plan_sha256=%s\n' "$qcap_model_plan_sha"
    printf 'weight_route_manifest_sha256=%s\n' "$qcap_route_sha"
    printf 'kernel_manifest_sha256=%s\n' "$qcap_manifest_sha"
    printf 'admitted_bootstrap_sha256=%s\n' "$qcap_bootstrap_sha"
    printf 'max_batch_rows=%s\n' "$qcap_max_batch_rows"
    printf 'max_query_rows=%s\n' "$qcap_max_query_rows"
    printf 'max_query_tokens=%s\n' "$qcap_max_query_tokens"
    printf 'max_concurrency=32\n'
    printf 'runtime_executable_sha256=%s\n' "$qcap_runtime_sha"
    printf 'token_id_sse_bridge_sha256=%s\n' "$qcap_bridge_sha"
  } >"$qcap_authentication_output"
}

qcap_write_receipt() {
  qcap_receipt_output=$1
  qcap_authentication_sha=$2
  printf '%s\n' \
    "{\"authenticated\":true,\"authentication_sha256\":\"$qcap_authentication_sha\",\"configuration_sha256\":\"$qcap_launch_sha\",\"max_concurrency\":32,\"model_content_sha256\":\"$qcap_model_content_sha\",\"runtime_executable_sha256\":\"$qcap_runtime_sha\",\"schema\":\"lunaflux.qwen3-authenticated-capacity.v1\",\"token_id_sse_bridge_sha256\":\"$qcap_bridge_sha\"}" \
    >"$qcap_receipt_output"
}
