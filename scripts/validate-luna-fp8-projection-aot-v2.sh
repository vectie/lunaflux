#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

moon check kernels/luna_cuda_fp8_projection_aot kernels/luna_kernel_bundle \
  --target native --deny-warn --warn-list +73
moon test kernels/luna_cuda_fp8_projection_aot kernels/luna_kernel_bundle \
  --target native --deny-warn --warn-list +73

if rg -n 'runtime/|engine/|cmd/|internal/cuda|extern[[:space:]]+"[cC]"' \
  kernels/luna_cuda_fp8_projection_aot kernels/luna_kernel_bundle \
  --glob '*.mbt' --glob 'moon.pkg'; then
  echo "FP8 v2 offline packages crossed into runtime or native authority" >&2
  exit 1
fi

rg -q 'lunaflux\.luna-cuda-fp8-projection-aot\.v2' \
  kernels/luna_cuda_fp8_projection_aot/v2_recipe.mbt
rg -q '_fp8_e4m3_w8a8_release_v2' \
  kernels/luna_cuda_fp8_projection_aot/v2_recipe.mbt
rg -q '__nv_cvt_float_to_fp8.*__NV_SATFINITE.*__NV_E4M3' \
  kernels/luna_cuda_fp8_projection_aot/v2_source.mbt
rg -q 'bounded == 0\.0f \? 0\.0f : bounded' \
  kernels/luna_cuda_fp8_projection_aot/v2_source.mbt
rg -q 'scale_row = 0; scale_row < token_count' \
  kernels/luna_cuda_fp8_projection_aot/v2_source.mbt
rg -q 'scale_column = 0; scale_column < ' \
  kernels/luna_cuda_fp8_projection_aot/v2_source.mbt
rg -q 'workspace_publication=serial_index_zero_after_full_output_validation_v2' \
  kernels/luna_cuda_fp8_projection_aot/v2_recipe.mbt

if rg -n '__syncthreads|cooperative_groups' \
  kernels/luna_cuda_fp8_projection_aot/v2_source.mbt; then
  echo "FP8 v2 reference source introduced an invalid grid synchronization" >&2
  exit 1
fi

cell_zero_failure_templates="$(rg -o 'workspace\[0\] = lf_v2_failure\(\)' \
  kernels/luna_cuda_fp8_projection_aot/v2_source.mbt | wc -l | tr -d ' ')"
cell_one_failure_templates="$(rg -o 'workspace\[1\] = lf_v2_failure\(\)' \
  kernels/luna_cuda_fp8_projection_aot/v2_source.mbt | wc -l | tr -d ' ')"
cell_zero_success_templates="$(rg -o 'workspace\[0\] = external_scale' \
  kernels/luna_cuda_fp8_projection_aot/v2_source.mbt | wc -l | tr -d ' ')"
cell_one_success_templates="$(rg -o 'workspace\[1\] = post_silu_scale' \
  kernels/luna_cuda_fp8_projection_aot/v2_source.mbt | wc -l | tr -d ' ')"
if [ "$cell_zero_failure_templates" -ne 1 ] || \
  [ "$cell_one_failure_templates" -ne 1 ] || \
  [ "$cell_zero_success_templates" -ne 3 ] || \
  [ "$cell_one_success_templates" -ne 1 ]; then
  echo "FP8 v2 workspace writer templates drifted" >&2
  exit 1
fi

rg -q 'admit_fp8_projection_compiled_operation_v2' \
  kernels/luna_kernel_bundle/fp8_projection.mbt
rg -q 'never external approval, executable authority, promotion, or readiness' \
  kernels/luna_kernel_bundle/types.mbt
if rg -n 'luna_kernel_bundle|LunaDeterministicCompileReceipt' \
  engine/fp8_device_executor kernels/fp8_release_authority \
  --glob '*.mbt' --glob 'moon.pkg'; then
  echo "caller-constructible compile evidence crossed executable admission" >&2
  exit 1
fi
rg -q 'non-identical second CUBIN' \
  kernels/luna_kernel_bundle/fp8_projection_v2_release_wbtest.mbt
rg -q 'artifact symbol substitution' \
  kernels/luna_kernel_bundle/fp8_projection_v2_release_wbtest.mbt
rg -q 'b88549dbdfd78a1bd601a4b3fb392c6bdcfb400df977b79405d0d529f11a53a3' \
  kernels/luna_cuda_fp8_projection_aot/lowering_test.mbt

while IFS= read -r source_file; do
  line_count="$(wc -l < "$source_file")"
  if [ "$line_count" -ge 500 ]; then
    echo "FP8 v2 source exceeds the under-500-line boundary: $source_file" >&2
    exit 1
  fi
done < <(find kernels/luna_cuda_fp8_projection_aot kernels/luna_kernel_bundle \
  -type f -name '*.mbt' -print)

echo "Luna FP8 projection gated AOT v2 and bundle boundary is valid."
