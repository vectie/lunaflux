#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

probe_root=tests/fused_parallel_cuda_probe

for anchor in \
  'vectie/lunaflux/release/luna_fused_candidate_export' \
  'vectie/lunaflux/kernels/luna_cuda_fused_parallel_aot' \
  'vectie/lunaflux/device' \
  'vectie/lunaflux/release/luna_fused_physical_evidence' \
  'vectie/lunaflux/tests/fused_parallel_qualification'; do
  rg -Fq "$anchor" "$probe_root/moon.pkg" || {
    echo "fused CUDA probe dependency is missing: $anchor" >&2
    exit 1
  }
done

for anchor in \
  'prepare_qualification_set' \
  'bind_fused_qkv_rope_kv_compiled_candidate' \
  'bind_fused_residual_rmsnorm_compiled_candidate' \
  '!qkv.manifest_bindable() && !residual.manifest_bindable()' \
  'compute_major() == 12' \
  'compute_minor() == 0'; do
  rg -Fq "$anchor" "$probe_root" || {
    echo "fused CUDA authentication anchor is missing: $anchor" >&2
    exit 1
  }
done

for anchor in \
  'single-origin' \
  'single-page-tail' \
  'cross-page-pair' \
  'eight-token-page-boundaries' \
  'qualify_qkv_bf16_output' \
  'qualify_residual_norm_bf16_output' \
  'compare_qkv_bf16_output' \
  'compare_residual_norm_bf16_output' \
  'maximum_absolute_error_ppb' \
  'maximum_relative_error_ppb' \
  'expected_dispatch_canary'; do
  rg -Fq "$anchor" "$probe_root" || {
    echo "fused CUDA referee anchor is missing: $anchor" >&2
    exit 1
  }
done

for anchor in \
  'try residual_function.close() catch' \
  'try residual_module.close() catch' \
  'try qkv_function.close() catch' \
  'try qkv_module.close() catch' \
  'try arena.close() catch' \
  'try stream.close() catch' \
  'try context.close() catch' \
  'balance.cleanup_failures += 1' \
  'device_bytes' \
  'pending_work' \
  'resources=\{balance.render()}'; do
  rg -Fq "$anchor" "$probe_root" || {
    echo "ordered cleanup/resource anchor is missing: $anchor" >&2
    exit 1
  }
done
if rg -n 'try! .*\.close\(\)' "$probe_root" --glob '*.mbt' >/dev/null; then
  echo 'fused CUDA probe cleanup can abort before later resources close' >&2
  exit 1
fi

if rg -ni 'nvrtc|--ptx|\.ptx|runtime.{0,16}compil|manifest_bindable\([^)]*true|promotion_evidence|runtime.serv' \
  --glob '*.mbt' --glob '*.sh' "$probe_root" \
  scripts/run-fused-parallel-physical-campaign.sh \
  >/dev/null; then
  echo 'fused CUDA probe introduced JIT, PTX, runtime, manifest, or promotion authority' >&2
  exit 1
fi

for anchor in \
  'scripts/build-luna-cuda-aot.sh' \
  'independent fused CUBIN publications differ' \
  'oracle_qkv' \
  'oracle_rope' \
  'oracle_kv' \
  '--tool memcheck' \
  '--tool racecheck' \
  '--tool initcheck' \
  '--leak-check full' \
  '--error-exitcode 99' \
  'ERROR SUMMARY: 0 errors'; do
  rg -Fq -- "$anchor" scripts/run-fused-parallel-physical-campaign.sh \
    scripts/fused-physical-campaign-functions.sh || {
    echo "physical runner anchor is missing: $anchor" >&2
    exit 1
  }
done

rg -Fq 'is retired: use the sealed run-fused-parallel-physical-campaign.sh' \
  scripts/probe-fused-parallel-cuda.sh || {
  echo 'obsolete unsealed fused helper is not explicitly retired' >&2
  exit 1
}

max_lines=$(find "$probe_root" -type f \( -name '*.mbt' -o -name '*.sh' \) \
  -exec wc -l {} + | awk '$2 != "total" {print $1}' | sort -nr | sed -n '1p')
[ "$max_lines" -lt 500 ] || {
  echo 'fused CUDA probe contains a source file at or above 500 lines' >&2
  exit 1
}

sh -n scripts/probe-fused-parallel-cuda.sh
bash -n scripts/run-fused-parallel-physical-campaign.sh
moon check --target native --deny-warn --warn-list +73 "$probe_root"
moon test --target native --deny-warn --warn-list +73 "$probe_root"
scripts/validate-fused-parallel-candidates.sh >/dev/null
scripts/test-luna-cuda-aot-builder.sh >/dev/null
scripts/validate-cuda-abi.sh >/dev/null
scripts/validate-cuda-ordered-executor-sanitizer.sh >/dev/null

echo 'LunaFlux fused parallel CUDA qualification probe boundary passed.'
