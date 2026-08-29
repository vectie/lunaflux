#!/bin/sh
set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"
runner=scripts/run-qwen3-bf16-physical-campaign.sh

fail() {
  printf 'Qwen3 BF16 physical validator failed: %s\n' "$1" >&2
  exit 1
}

bash -n "$runner" || fail 'runner syntax is invalid'
for anchor in \
  'qkv_independent_widths=passed' \
  'qk_rms_norm_capability=11' \
  'build-luna-bf16-kernel-set.sh' \
  'verify-luna-bf16-kernel-set.sh' \
  'qwen3-numeric-materialization-pass' \
  'native_framed_c1_serving=separate-qwen-v12-physical-campaign-required' \
  'openai_sse_benchmark=blocked-authenticated-qwen-token-id-sse-bridge-unavailable' \
  'standard_openai_responses_profile_satisfies_benchmark_adapter=false' \
  'release_bind_max_batch_rows_native=1' \
  'release_bind_max_query_rows_native=1' \
  'benchmark_c32=separate-authenticated-release-profile-required' \
  'release_bind_max_batch_rows_benchmark=32' \
  'release_bind_max_query_rows_benchmark=32' \
  'engine_server_gpu_concurrency=exactly-one-required' \
  'lunaflux_seal_evidence_directory'; do
  grep -F "$anchor" "$runner" >/dev/null || fail "runner anchor is absent: $anchor"
done
if grep -E -i 'llama|mistral' "$runner" >/dev/null; then
  fail 'Qwen-only runner contains another model family'
fi
. scripts/luna-bf16-kernel-producer-common.sh
lbf_is_family qk_rms_norm || fail 'shared producer rejects qk_rms_norm'
grep -F 'embedding_lookup|rms_norm|qk_rms_norm|positioned_rotary|residual_add)' \
  scripts/build-luna-bf16-kernel-set.sh >/dev/null ||
  fail 'offline builder does not route qk_rms_norm through pointwise compilation'
if bash "$runner" >/dev/null 2>&1; then
  fail 'runner accepted missing arguments'
else
  status=$?
  [ "$status" -eq 2 ] || fail 'runner missing-argument status drifted'
fi
moon check tests/qwen3_bf16_physical --target native --deny-warn --warn-list +73
moon test tests/qwen3_bf16_physical --target native --deny-warn --warn-list +73
printf '%s\n' 'Qwen3 BF16 physical campaign static validation passed'
