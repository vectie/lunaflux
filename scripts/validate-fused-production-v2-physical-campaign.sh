#!/usr/bin/env bash
set -euo pipefail
LC_ALL=C
export LC_ALL
[[ $# == 0 || ( $# == 1 && $1 == --static-only ) ]] || { printf 'usage: %s [--static-only]\n' "$0" >&2; exit 2; }
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$root"
fail() { printf 'fused production V2 boundary rejected: %s\n' "$1" >&2; exit 1; }
probe=tests/fused_parallel_cuda_probe
for anchor in \
  'lower_fused_qkv_rope_paged_kv_write_production_candidate' \
  'lower_fused_residual_rmsnorm_production_candidate' \
  'lower_paged_attention_readonly_production_cuda_aot_candidate' \
  'bind_fused_qkv_rope_paged_kv_write_production_compiled_candidate' \
  'bind_fused_residual_rmsnorm_production_compiled_candidate' \
  'OrderedKernelCaptureRequired' 'OrderedKernelCaptured' \
  'probe_production_qkv_arguments' 'probe_attention_arguments' \
  'probe_production_residual_arguments' 'oracle_attention' \
  'manifest_bindable=0 promotion_authority=0'; do
  rg -Fq "$anchor" "$probe" || fail "probe anchor is absent: $anchor"
done
for anchor in \
  'QkvPositionedRopePagedKvWriteProductionAbiV2' \
  'RotatedQueryPagedKvReadOnlyProductionRawPointerV2' \
  'fused_qkv_production_argument_count() -> Int' \
  'readonly_attention_production_argument_count() -> Int' \
  'prepare_fused_qkv_readonly_steps'; do
  rg -Fq "$anchor" engine/device_step/fused_qkv_readonly_*.mbt || fail "serving QKV/readonly join drifted: $anchor"
done
for anchor in 'ResidualRmsNormProductionAbiV2 => 6' 'prepare_fused_residual_step'; do
  rg -Fq "$anchor" engine/device_step/fused_grouped_*.mbt || fail "serving residual join drifted: $anchor"
done
for anchor in \
  'validate-fused-physical-approved-policy.sh' \
  'export-production' 'run-production' \
  'production_qkv_v2' 'production_readonly_attention_v2' \
  'production_residual_rmsnorm_v2' 'oracle_attention' \
  'run_sanitizer memcheck' 'run_sanitizer racecheck' 'run_sanitizer initcheck' \
  'FILES.sha256' 'RESULT.txt' 'OUTER_SEAL.sha256' \
  'SPAWN_JOIN.v1' 'aggregate_runtime=externally-prepared-required' \
  'normal_spawn_admission=not-claimed' \
  'qualification_only=true' 'promotion_authority=absent' \
  'verify-fused-production-v2-physical-campaign.sh'; do
  rg -Fq "$anchor" scripts/run-fused-production-v2-physical-campaign.sh || fail "campaign anchor is absent: $anchor"
done
for anchor in 'spawn join record is absent or substituted' \
  'aggregate_runtime) == externally-prepared-required' \
  'normal_spawn_admission) == not-claimed'; do
  rg -Fq "$anchor" scripts/verify-fused-production-v2-physical-campaign.sh ||
    fail "spawn join verifier anchor is absent: $anchor"
done
if rg -ni 'nvrtc|--ptx|\.ptx|promotion_authority=(true|granted)|manifest_bindable=(true|1)' \
  "$probe" scripts/run-fused-production-v2-physical-campaign.sh \
  scripts/verify-fused-production-v2-physical-campaign.sh >/dev/null; then
  fail 'probe gained JIT, PTX, manifest, or promotion authority'
fi
max_lines=$(find "$probe" -type f -name '*.mbt' -exec wc -l {} + | awk '$2 != "total" {print $1}' | sort -nr | sed -n '1p')
[[ $max_lines -lt 500 ]] || fail 'probe contains a MoonBit file at or above 500 lines'
for file in scripts/run-fused-production-v2-physical-campaign.sh \
  scripts/verify-fused-production-v2-physical-campaign.sh \
  scripts/test-fused-production-v2-evidence-verifier.sh; do
  [[ $(wc -l <"$file" | tr -d ' ') -lt 500 ]] || fail "$file is at or above 500 lines"
  bash -n "$file" || fail "$file has invalid shell syntax"
done
rg -Fq 'is retired: use the sealed run-fused-parallel-physical-campaign.sh' \
  scripts/probe-fused-parallel-cuda.sh || fail 'stale unsealed helper was not retired'
sh -n scripts/probe-fused-parallel-cuda.sh
moon check --target native --deny-warn --warn-list +73 "$probe"
moon test --target native --deny-warn --warn-list +73 "$probe"
moon test --target native --deny-warn --warn-list +73 kernels/luna_cuda_fused_parallel_aot
moon test --target native --deny-warn --warn-list +73 kernels/luna_cuda_paged_attention_readonly_aot
scripts/test-fused-production-v2-evidence-verifier.sh >/dev/null
moon build tests/fused_parallel_cuda_probe --target native --deny-warn --warn-list +73 >/dev/null
exe=$root/_build/native/debug/build/tests/fused_parallel_cuda_probe/fused_parallel_cuda_probe.exe
work=$(mktemp -d "${TMPDIR:-/tmp}/lunaflux-fused-v2-export-test.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
"$exe" export-production "$(printf 'b%.0s' {1..64})" "$work/export" >/dev/null 2>"$work/first.stderr"
[[ ! -s $work/first.stderr ]] || fail 'local production exporter emitted stderr'
if "$exe" export-production "$(printf 'b%.0s' {1..64})" "$work/export" >/dev/null 2>"$work/second.stderr"; then
  fail 'production exporter overwrote an existing output'
fi
source_sha=$(sed -n 's/^production_qkv_source_sha256=//p' "$work/export/IDENTITIES.v1")
if command -v sha256sum >/dev/null 2>&1; then observed=$(sha256sum "$work/export/sources/production_qkv_v2.cu" | awk '{print $1}')
else observed=$(shasum -a 256 "$work/export/sources/production_qkv_v2.cu" | awk '{print $1}'); fi
[[ $source_sha == "$observed" ]] || fail 'exported source identity is not deterministic'
echo 'LunaFlux fused production V2 physical campaign boundary passed (GPU not executed).'
