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
rg -F '@json_reader.parse_bytes' \
  cmd/lunaflux_qwen3_token_id_bridge/startup.mbt >/dev/null ||
  fail 'startup tokenizer materialization is absent'
rg -F 'Qwen3IncrementalDetokenizer' \
  benchmarks/qwen3_token_id_bridge cmd/lunaflux_qwen3_token_id_bridge \
  >/dev/null || fail 'incremental Qwen detokenization is absent'
rg -F 'lunaflux.benchmark-terminal.v1\",\"text\"' \
  benchmarks/qwen3_token_id_bridge/sse.mbt >/dev/null ||
  fail 'terminal stabilized text is absent'
if rg -n '@fs|@crypto|sha256|parse_bytes' \
  benchmarks/qwen3_token_id_bridge/detokenize.mbt \
  benchmarks/qwen3_token_id_bridge/sse.mbt \
  cmd/lunaflux_qwen3_token_id_bridge/server.mbt; then
  fail 'token streaming performs filesystem, identity, or tokenizer parsing work'
fi
if rg -n 'Llama|Mistral|/v1/responses|python|PyTorch|subprocess|posix_spawn' \
  benchmarks/qwen3_token_id_bridge cmd/lunaflux_qwen3_token_id_bridge \
  scripts/start-qwen3-lunaflux-token-id-bridge.sh; then
  fail 'foreign-family, Python, Responses, or process-spawn fallback is present'
fi

moon check --target native --deny-warn --warn-list +73 \
  benchmarks/qwen3_token_id_bridge cmd/lunaflux_qwen3_token_id_bridge
moon test --target native --deny-warn --warn-list +73 \
  benchmarks/qwen3_token_id_bridge cmd/lunaflux_qwen3_token_id_bridge
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git diff --check -- \
    benchmarks/qwen3_token_id_bridge cmd/lunaflux_qwen3_token_id_bridge \
    scripts/start-qwen3-lunaflux-token-id-bridge.sh \
    scripts/validate-qwen3-token-id-bridge.sh
fi

printf '%s\n' 'qwen3 token-ID bridge boundary passed'
