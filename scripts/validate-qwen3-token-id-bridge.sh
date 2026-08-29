#!/bin/sh
set -eu

fail() {
  printf 'qwen3 token-ID bridge boundary failed: %s\n' "$1" >&2
  exit 1
}

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"

for path in \
  benchmarks/qwen3_token_id_bridge \
  cmd/lunaflux_qwen3_token_id_bridge \
  scripts/start-qwen3-lunaflux-token-id-bridge.sh; do
  [ -e "$path" ] || fail "missing $path"
done

rg -F 'Qwen3-0.6B' benchmarks/qwen3_token_id_bridge \
  cmd/lunaflux_qwen3_token_id_bridge >/dev/null || fail 'Qwen model pin is absent'
rg -F '/benchmark/v1/token-ids' cmd/lunaflux_qwen3_token_id_bridge >/dev/null ||
  fail 'benchmark endpoint is absent'
rg -F 'native-framed-v1' cmd/lunaflux_qwen3_token_id_bridge >/dev/null ||
  fail 'native-framed upstream is absent'
if rg -n 'Llama|Mistral|/v1/responses|python|PyTorch|subprocess|posix_spawn' \
  benchmarks/qwen3_token_id_bridge cmd/lunaflux_qwen3_token_id_bridge \
  scripts/start-qwen3-lunaflux-token-id-bridge.sh; then
  fail 'foreign-family, Python, Responses, or process-spawn fallback is present'
fi

moon check --target native --deny-warn --warn-list +73 \
  benchmarks/qwen3_token_id_bridge cmd/lunaflux_qwen3_token_id_bridge
moon test --target native --deny-warn --warn-list +73 \
  benchmarks/qwen3_token_id_bridge cmd/lunaflux_qwen3_token_id_bridge
git diff --check -- \
  benchmarks/qwen3_token_id_bridge cmd/lunaflux_qwen3_token_id_bridge \
  scripts/start-qwen3-lunaflux-token-id-bridge.sh

printf '%s\n' 'qwen3 token-ID bridge boundary passed'
