#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"
materializer=scripts/materialize-qwen3-bf16-v12-launch.sh
serving_contract=docs/QWEN3_V12_SERVING.md
release_binder=cmd/lunaflux_qwen3_bf16_release_bind/main.mbt

fail() {
  printf '%s\n' "Qwen3 v12 serving validator failed: $1" >&2
  exit 1
}

sh -n "$materializer" || fail 'materializer syntax is invalid'
for wiring in "$materializer" scripts/augment-luna-kernel-root-plan-with-fused-runtime.sh; do
  sh -n "$wiring" || fail 'fused-runtime wiring syntax is invalid'
  grep -F 'schema=lunaflux-reusable-fused-runtime-bundle.v3|schema=lunaflux-reusable-fused-runtime-bundle.v4)' "$wiring" >/dev/null ||
    fail 'fused-runtime wiring must accept legacy v3 and compiler-metadata v4'
done
for anchor in \
  'lunaflux.runtime.qwen3_bf16.v1' \
  'lunaflux.runtime.qwen3_bf16.v2' \
  'dense_qwen3_bf16_paged_aot_v12' \
  'augment-luna-kernel-root-plan-with-fused-runtime.sh' \
  'reusable-fused-residual.runtime.v1' \
  'reusable-fused-runtime-bundle.v3' \
  'authenticated_embedded_greedy_sampling' \
  'sampling_runtime=host' \
  'validate-release' \
  'serve-check'; do
  grep -R -F "$anchor" "$materializer" tests/qwen3_bf16_physical >/dev/null ||
    fail "serving anchor is absent: $anchor"
done
if grep -F 'cp "$fused_residual"' "$materializer" >/dev/null; then
  fail 'reusable fused residual runtime bypasses kernel-root plan assembly'
fi
for anchor in \
  'native-framed-v1|native-framed-c32-benchmark-v1|openai-responses-v1' \
  'native correctness profile requires authenticated c1 release geometry' \
  'native-framed-c32-benchmark-v1|openai-responses-v1' \
  'policy_schema=lunaflux.instance-policy.v3' \
  '"mode":"openai_responses_v1"' \
  '"invocation_path":"/v1/responses"' \
  '"model_alias":"qwen3-0.6b-bf16"' \
  '"system_prefix":"<|im_start|>system\\n"' \
  '"user_prefix":"<|im_start|>user\\n"' \
  '"assistant_cue":"<|im_start|>assistant\\n"' \
  'benchmark profile requires authenticated c32 release geometry' \
  'release_max_batch_rows=$(bind_value max_batch_rows)' \
  'release_max_query_rows=$(bind_value max_query_rows)' \
  'release_max_query_tokens=$(bind_value max_query_tokens)' \
  'scheduler_max_active_requests=$release_max_batch_rows' \
  'scheduler_max_waiting_requests=$release_max_batch_rows' \
  'materialization_profile=%s' \
  'external_protocol=%s'; do
  grep -F "$anchor" "$materializer" >/dev/null ||
    fail "Qwen OpenAI profile anchor is absent: $anchor"
done
for anchor in \
  '"max_batch_rows="' \
  '"max_query_rows="' \
  '"max_query_tokens="'; do
  grep -F "$anchor" "$release_binder" >/dev/null ||
    fail "authenticated Qwen release geometry is absent: $anchor"
done
if grep -E -i 'llama|mistral' "$materializer" >/dev/null; then
  fail 'Qwen-only materializer contains another family'
fi
if sh "$materializer" >/dev/null 2>&1; then
  fail 'materializer accepted missing arguments'
else
  status=$?
  [ "$status" -eq 2 ] || fail 'missing-argument status drifted'
fi
hostile_output=$(sh "$materializer" x x x x x x x x x x x 1 /tmp/qwen-v12-hostile unsupported-profile 2>&1) &&
  fail 'materializer accepted an unsupported profile'
printf '%s\n' "$hostile_output" |
  grep -F 'materialization profile is unsupported' >/dev/null ||
  fail 'unsupported profile did not fail at the closed profile boundary'
scripts/validate-qwen3-capacity-receipt.sh
for anchor in \
  'Fixed descriptor 5' \
  'LFD1DRN' \
  'Fixed descriptor 6' \
  'LFC1KEY' \
  'native-framed-v1' \
  'native-framed-c32-benchmark-v1' \
  'openai-responses-v1' \
  'authenticated c32' \
  '/v1/responses'; do
  grep -F "$anchor" "$serving_contract" >/dev/null ||
    fail "Qwen serving contract anchor is absent: $anchor"
done
moon check tests/qwen3_bf16_physical --target native --deny-warn --warn-list +73
moon test tests/qwen3_bf16_physical --target native --deny-warn --warn-list +73
printf '%s\n' 'Qwen3 v12 serving wiring validation passed'
