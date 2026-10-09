#!/bin/sh
set -eu
LC_ALL=C
export LC_ALL
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)

fail() {
  printf 'Qwen3 llama.cpp benchmark server rejected: %s\n' "$1" >&2
  exit 2
}

[ "$#" -eq 7 ] || fail 'usage: ABS_LLAMA_CPP_ROOT EXPECTED_VERSION ABS_MODEL_ROOT ABS_MODEL_ADMISSION#sha256=HEX HOST PORT EXPECTED_GGUF_SHA256'
environment=$1
expected_version=$2
model_root=$3
admission_argument=$4
host=$5
port=$6
expected_gguf_sha256=$7

case "$environment" in /*) ;; *) fail 'llama.cpp root must be absolute' ;; esac
[ -d "$environment" ] && [ ! -L "$environment" ] || fail 'llama.cpp root is unavailable'
[ "$(CDPATH= cd -- "$environment" && pwd -P)" = "$environment" ] || fail 'llama.cpp root is not canonical'
server=$environment/build-cuda/bin/llama-server
model=$environment/models/Qwen3-0.6B-BF16.gguf
[ -x "$server" ] && [ ! -L "$server" ] || fail 'llama-server is unavailable'
[ -f "$model" ] && [ ! -L "$model" ] || fail 'Qwen BF16 GGUF is unavailable'
[ "$host" = 127.0.0.1 ] || fail 'server must bind loopback'
case "$port" in ''|*[!0-9]*) fail 'port is not decimal' ;; esac
[ "$port" -ge 1024 ] && [ "$port" -le 65535 ] || fail 'port is outside bounds'
case "$expected_gguf_sha256" in *[!0-9a-f]*|'') fail 'GGUF SHA-256 is invalid' ;; esac
[ "${#expected_gguf_sha256}" -eq 64 ] || fail 'GGUF SHA-256 is invalid'
case "$admission_argument" in /*#sha256=*) ;; *) fail 'model admission must be digest suffixed' ;; esac
[ -d "$model_root" ] && [ ! -L "$model_root" ] || fail 'model root is unavailable'
[ "$(CDPATH= cd -- "$model_root" && pwd -P)" = "$model_root" ] || fail 'model root is not canonical'
/usr/bin/python3 -B "$repo_root/benchmarks/qwen3_comparison/verify_model_admission.py" \
  "$model_root" "$admission_argument" || fail 'campaign model admission mismatch'
grep -Eq '"model_type"[[:space:]]*:[[:space:]]*"qwen3"' "$model_root/config.json" ||
  fail 'model config is not Qwen3'

observed_version=$("$server" --version 2>&1 | sed -n 's/^version: //p') ||
  fail 'llama.cpp version is unavailable'
[ "$observed_version" = "$expected_version" ] || fail 'llama.cpp version pin mismatch'
observed_gguf_sha256=$(sha256sum "$model" | awk '{print $1}') || fail 'GGUF digest failed'
[ "$observed_gguf_sha256" = "$expected_gguf_sha256" ] || fail 'GGUF digest mismatch'

exec "$server" \
  --model "$model" \
  --alias Qwen3-0.6B \
  --host "$host" \
  --port "$port" \
  --ctx-size 65536 \
  --parallel 32 \
  --cont-batching \
  --flash-attn on \
  --cache-type-k bf16 \
  --cache-type-v bf16 \
  --n-gpu-layers 99 \
  --cache-reuse 0 \
  --no-webui
