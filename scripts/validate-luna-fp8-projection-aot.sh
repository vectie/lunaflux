#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

moon check kernels/luna_cuda_fp8_projection_aot kernels/luna_kernel_bundle \
  kernels/numeric_capability_manifest kernels/fp8_launch_abi \
  --target native --deny-warn
moon test kernels/luna_cuda_fp8_projection_aot kernels/luna_kernel_bundle \
  kernels/numeric_capability_manifest kernels/fp8_launch_abi \
  --target native --deny-warn

if rg -n 'runtime/|engine/|cmd/' kernels/luna_cuda_fp8_projection_aot; then
  echo "FP8 AOT package crossed into runtime, engine, or CLI authority" >&2
  exit 1
fi

rg -q '#include <cuda_fp8.h>' \
  kernels/luna_cuda_fp8_projection_aot/source_common.mbt
rg -q '__nv_cvt_float_to_fp8.*__NV_SATFINITE.*__NV_E4M3' \
  kernels/luna_cuda_fp8_projection_aot/source_common.mbt
rg -q '__CUDA_ARCH__ != 890' \
  kernels/luna_cuda_fp8_projection_aot/source_common.mbt
rg -q '__CUDA_ARCH__ != 900' \
  kernels/luna_cuda_fp8_projection_aot/source_common.mbt
rg -q 'if \(index == 0ULL\).*activation_scale_workspace' \
  kernels/luna_cuda_fp8_projection_aot/source_common.mbt
rg -q 'CompoundGatedActivationScale' \
  kernels/luna_cuda_fp8_projection_aot/validation.mbt

while IFS= read -r source_file; do
  line_count="$(wc -l < "$source_file")"
  if (( line_count >= 500 )); then
    echo "FP8 AOT source exceeds the under-500-line boundary: $source_file" >&2
    exit 1
  fi
done < <(find kernels/luna_cuda_fp8_projection_aot -type f \( -name '*.mbt' -o -name '*.md' \) -print)

echo "Luna FP8 projection offline AOT boundary is valid."
