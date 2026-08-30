#!/bin/sh
set -eu
LC_ALL=C
export LC_ALL
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)

fail() {
  printf 'Qwen3 LunaFlux benchmark server rejected: %s\n' "$1" >&2
  exit 2
}

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | sed 's/[[:space:]].*$//'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | sed 's/[[:space:]].*$//'
  else
    fail 'no SHA-256 utility is available'
  fi
}

identity_path=''
identity_sha=''
verify_file_identity() {
  label=$1
  identity=$2
  case "$identity" in
    /*'#sha256='????????????????????????????????????????????????????????????????) ;;
    *) fail "$label identity is malformed" ;;
  esac
  identity_path=${identity%#sha256=*}
  identity_sha=${identity##*#sha256=}
  [ -f "$identity_path" ] && [ ! -L "$identity_path" ] ||
    fail "$label is not a regular file"
  canonical=$(CDPATH= cd -- "$(dirname -- "$identity_path")" && pwd -P)/$(basename -- "$identity_path")
  [ "$canonical" = "$identity_path" ] || fail "$label path is not canonical"
  [ "$(hash_file "$identity_path")" = "$identity_sha" ] || fail "$label digest mismatch"
}

[ "$#" -eq 21 ] || fail \
  'usage: NATIVE EXACT_REVISION_SHA MODEL_ROOT MODEL_ADMISSION HOST PORT RUNTIME#sha SUPERVISOR#sha BRIDGE#sha DEPLOYMENT#sha TOKENIZER#sha LAUNCH#sha RELEASE_BIND#sha CAPACITY#sha RUNTIME_ADDR MODEL_CONTENT_SHA MODEL_PLAN_SHA MAX_INPUT MAX_OUTPUT MAX_TOKEN_ID MAX_CONTEXT'
environment=$1
expected_version=$2
model_root=$3
admission_argument=$4
host=$5
port=$6
runtime_identity=$7
supervisor_identity=$8
bridge_identity=$9
shift 9
deployment_identity=$1
tokenizer_identity=$2
launch_identity=$3
release_bind_identity=$4
capacity_identity=$5
runtime_address=$6
model_content_sha=$7
model_plan_sha=$8
max_input=$9
shift 9
max_output=$1
max_token_id=$2
max_context=$3

[ "$environment" = native ] || fail 'LunaFlux environment must be native'
case "$expected_version" in ''|*[!0123456789abcdef]*) fail 'LunaFlux revision identity is not lowercase SHA-256' ;; esac
[ "${#expected_version}" -eq 64 ] || fail 'LunaFlux revision identity is not lowercase SHA-256'
[ "$host" = 127.0.0.1 ] || fail 'bridge must bind loopback'
case "$port" in ''|*[!0-9]*) fail 'bridge port is not decimal' ;; esac
[ "$port" -ge 1024 ] && [ "$port" -le 65535 ] || fail 'bridge port is outside bounds'
case "$runtime_address" in 127.0.0.1:*) ;; *) fail 'native runtime address must be exact loopback' ;; esac
runtime_port=${runtime_address#127.0.0.1:}
case "$runtime_port" in ''|*[!0-9]*) fail 'native runtime port is not decimal' ;; esac
[ "$runtime_port" -ge 1024 ] && [ "$runtime_port" -le 65535 ] || fail 'native runtime port is outside bounds'
[ "$runtime_port" != "$port" ] || fail 'bridge and native runtime ports must differ'
for value in "$model_content_sha" "$model_plan_sha"; do
  case "$value" in ''|*[!0123456789abcdef]*) fail 'model digest is malformed' ;; esac
  [ "${#value}" -eq 64 ] || fail 'model digest is malformed'
done
for value in "$max_input" "$max_output" "$max_token_id" "$max_context"; do
  case "$value" in ''|*[!0-9]*|0) fail 'token/context limit is invalid' ;; esac
done
[ $((max_input + max_output)) -le "$max_context" ] || fail 'benchmark limits exceed native context capacity'

[ -d "$model_root" ] && [ ! -L "$model_root" ] || fail 'model root is unavailable'
[ "$(CDPATH= cd -- "$model_root" && pwd -P)" = "$model_root" ] || fail 'model root is not canonical'
python3 -B "$repo_root/benchmarks/qwen3_comparison/verify_model_admission.py" \
  "$model_root" "$admission_argument" || fail 'campaign model admission mismatch'

verify_file_identity 'native runtime executable' "$runtime_identity"
runtime_executable=$identity_path
[ -x "$runtime_executable" ] || fail 'native runtime is not executable'
verify_file_identity 'native supervisor executable' "$supervisor_identity"
supervisor_executable=$identity_path
[ -x "$supervisor_executable" ] || fail 'native supervisor is not executable'
verify_file_identity 'token-ID bridge executable' "$bridge_identity"
bridge_executable=$identity_path
[ -x "$bridge_executable" ] || fail 'token-ID bridge is not executable'
verify_file_identity 'tokenizer JSON' "$tokenizer_identity"
tokenizer_json=$identity_path
[ "$tokenizer_json" = "$model_root/tokenizer.json" ] || fail 'bridge tokenizer is not the admitted model tokenizer'
verify_file_identity 'native launch file' "$launch_identity"
launch_file=$identity_path
launch_sha=$identity_sha
verify_file_identity 'release-bind receipt' "$release_bind_identity"
verify_file_identity 'capacity receipt' "$capacity_identity"

case "$deployment_identity" in /*'#sha256='*) ;; *) fail 'deployment identity is malformed' ;; esac
deployment_root=${deployment_identity%#sha256=*}
deployment_sha=${deployment_identity##*#sha256=}
[ -d "$deployment_root" ] && [ ! -L "$deployment_root" ] || fail 'native deployment is not a regular directory'
[ "$(CDPATH= cd -- "$deployment_root" && pwd -P)" = "$deployment_root" ] || fail 'native deployment path is not canonical'
[ "$launch_file" = "$deployment_root/lunaflux.launch.json" ] || fail 'native launch file is not owned by deployment root'
[ "$deployment_sha" = "$launch_sha" ] || fail 'deployment launch digest differs'
"$repo_root/scripts/verify-qwen3-authenticated-capacity.sh" \
  "$release_bind_identity" "$deployment_identity" "$runtime_identity" \
  "$bridge_identity" "$capacity_identity" >/dev/null ||
  fail 'authenticated c32 capacity receipt differs from its exact authorities'

stage=$(mktemp -d /tmp/lunaflux-qwen3-combined.XXXXXX)
runtime_stdout=$stage/runtime.stdout
runtime_stderr=$stage/runtime.stderr
supervisor_stdout=$stage/supervisor.stdout
supervisor_stderr=$stage/supervisor.stderr
bridge_stdout=$stage/bridge.stdout
bridge_stderr=$stage/bridge.stderr
drain_trigger=$stage/drain.request
supervisor_pid=''
bridge_wrapper_pid=''
shutting_down=0

relay_logs() {
  [ ! -s "$runtime_stdout" ] || sed 's/^/[native stdout] /' "$runtime_stdout"
  [ ! -s "$supervisor_stdout" ] || sed 's/^/[supervisor stdout] /' "$supervisor_stdout"
  [ ! -s "$bridge_stdout" ] || sed 's/^/[bridge stdout] /' "$bridge_stdout"
  [ ! -s "$runtime_stderr" ] || sed 's/^/[native stderr] /' "$runtime_stderr" >&2
  [ ! -s "$supervisor_stderr" ] || sed 's/^/[supervisor stderr] /' "$supervisor_stderr" >&2
  [ ! -s "$bridge_stderr" ] || sed 's/^/[bridge stderr] /' "$bridge_stderr" >&2
}

cleanup_stage() {
  rm -f "$runtime_stdout" "$runtime_stderr" "$supervisor_stdout" \
    "$supervisor_stderr" "$bridge_stdout" "$bridge_stderr" "$drain_trigger"
  rmdir "$stage" 2>/dev/null || true
}

shutdown() {
  [ "$shutting_down" -eq 0 ] || return
  shutting_down=1
  trap - TERM INT HUP
  if [ -n "$bridge_wrapper_pid" ] && kill -0 "$bridge_wrapper_pid" 2>/dev/null; then
    kill -TERM "$bridge_wrapper_pid" 2>/dev/null || true
    wait "$bridge_wrapper_pid" 2>/dev/null || true
  fi
  : >"$drain_trigger"
  supervisor_status=0
  if [ -n "$supervisor_pid" ]; then
    wait "$supervisor_pid" || supervisor_status=$?
  fi
  relay_logs
  cleanup_stage
  return "$supervisor_status"
}

on_signal() {
  status=0
  shutdown || status=$?
  exit "$status"
}
trap on_signal TERM INT HUP

"$supervisor_executable" "$runtime_executable" "$deployment_identity" \
  "$runtime_stdout" "$runtime_stderr" "$drain_trigger" \
  >"$supervisor_stdout" 2>"$supervisor_stderr" &
supervisor_pid=$!

ready=0
attempt=0
while [ "$attempt" -lt 6000 ]; do
  kill -0 "$supervisor_pid" 2>/dev/null || break
  if [ -f "$runtime_stdout" ] &&
    grep -Fxq 'readiness: true' "$runtime_stdout" &&
    grep -Fxq 'runtime_protocol=native-framed-v1' "$runtime_stdout" &&
    grep -Fxq "runtime_origin=tcp://$runtime_address" "$runtime_stdout"; then
    kill -0 "$supervisor_pid" 2>/dev/null || break
    ready=1
    break
  fi
  attempt=$((attempt + 1))
  sleep 0.1
done
if [ "$ready" -ne 1 ]; then
  shutdown || true
  fail 'native runtime did not publish exact readiness before bridge startup'
fi
if [ "$(grep -Fxc 'readiness: true' "$runtime_stdout")" -ne 1 ]; then
  shutdown || true
  fail 'native readiness publication is duplicated'
fi
if [ "$(grep -Fxc "runtime_origin=tcp://$runtime_address" "$runtime_stdout")" -ne 1 ]; then
  shutdown || true
  fail 'native runtime origin publication is duplicated'
fi

(
  bridge_child=''
  close_bridge() {
    trap - TERM INT HUP
    if [ -n "$bridge_child" ] && kill -0 "$bridge_child" 2>/dev/null; then
      kill -TERM "$bridge_child" 2>/dev/null || true
      wait "$bridge_child" 2>/dev/null || true
    fi
    : >"$drain_trigger"
  }
  trap 'close_bridge; exit 0' TERM INT HUP
  "$bridge_executable" "$bridge_identity" "$tokenizer_identity" "$launch_identity" \
    "$capacity_identity" "$host:$port" "$runtime_address" "$model_content_sha" \
    "$model_plan_sha" "$max_input" "$max_output" "$max_token_id" "$max_context" \
    >"$bridge_stdout" 2>"$bridge_stderr" &
  bridge_child=$!
  bridge_status=0
  wait "$bridge_child" || bridge_status=$?
  : >"$drain_trigger"
  [ "$bridge_status" -ne 0 ] || bridge_status=1
  exit "$bridge_status"
) &
bridge_wrapper_pid=$!

# The bridge wrapper requests a native drain if the bridge dies. Conversely,
# any native/supervisor failure wakes this leader, which then closes the bridge.
supervisor_status=0
wait "$supervisor_pid" || supervisor_status=$?
supervisor_pid=''
bridge_status=0
if kill -0 "$bridge_wrapper_pid" 2>/dev/null; then
  kill -TERM "$bridge_wrapper_pid" 2>/dev/null || true
fi
wait "$bridge_wrapper_pid" || bridge_status=$?
bridge_wrapper_pid=''
relay_logs
cleanup_stage
[ "$supervisor_status" -eq 0 ] || exit "$supervisor_status"
[ "$bridge_status" -eq 0 ] || exit "$bridge_status"
exit 1
