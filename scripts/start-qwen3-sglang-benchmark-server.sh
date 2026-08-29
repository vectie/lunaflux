#!/bin/sh
set -eu
LC_ALL=C
export LC_ALL
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)

fail() {
  printf 'Qwen3 SGLang benchmark server rejected: %s\n' "$1" >&2
  exit 2
}

[ "$#" -eq 6 ] || fail 'usage: ABS_ENV_PREFIX EXPECTED_SGLANG_VERSION ABS_MODEL_ROOT ABS_MODEL_ADMISSION#sha256=HEX HOST PORT'
environment=$1
expected_version=$2
model_root=$3
admission_argument=$4
host=$5
port=$6

case "$environment" in /*) ;; *) fail 'Conda environment prefix must be absolute' ;; esac
[ -d "$environment" ] && [ ! -L "$environment" ] || fail 'Conda environment prefix is unavailable'
[ "$(CDPATH= cd -- "$environment" && pwd -P)" = "$environment" ] || fail 'Conda environment prefix is not canonical'
[ -x "$environment/bin/python" ] || fail 'Conda environment Python is unavailable'
[ "$host" = 127.0.0.1 ] || fail 'server must bind loopback'
case "$port" in ''|*[!0-9]*) fail 'port is not decimal' ;; esac
[ "$port" -ge 1024 ] && [ "$port" -le 65535 ] || fail 'port is outside bounds'
case "$admission_argument" in /*#sha256=*) ;; *) fail 'model admission must be digest suffixed' ;; esac
[ -d "$model_root" ] && [ ! -L "$model_root" ] || fail 'model root is unavailable'
[ "$(CDPATH= cd -- "$model_root" && pwd -P)" = "$model_root" ] || fail 'model root is not canonical'
python3 -B "$repo_root/benchmarks/qwen3_comparison/verify_model_admission.py" \
  "$model_root" "$admission_argument" || fail 'campaign model admission mismatch'
grep -Eq '"model_type"[[:space:]]*:[[:space:]]*"qwen3"' "$model_root/config.json" ||
  fail 'model config is not Qwen3'

observed_version=$("$environment/bin/python" -c \
  'import importlib.metadata; print(importlib.metadata.version("sglang"))') ||
  fail 'SGLang Conda environment is unavailable'
[ "$observed_version" = "$expected_version" ] || fail 'SGLang version pin mismatch'

exec "$environment/bin/python" -m sglang.launch_server \
  --model-path "$model_root" \
  --tokenizer-path "$model_root" \
  --served-model-name Qwen3-0.6B \
  --host "$host" \
  --port "$port" \
  --dtype bfloat16 \
  --context-length 40960 \
  --tp-size 1 \
  --schedule-policy fcfs \
  --kv-cache-dtype auto \
  --max-running-requests 32 \
  --stream-interval 1 \
  --disable-radix-cache
