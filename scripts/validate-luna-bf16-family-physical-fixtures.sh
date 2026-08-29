#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd -P)
pointwise="$repo_root/kernels/luna_cuda_pointwise_aot/fixtures/physical_sm120"
projection="$repo_root/kernels/luna_cuda_projection_aot/fixtures/physical_sm120"

hash_check() {
  directory=$1
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "$directory" && sha256sum --check --strict SHA256SUMS)
  else
    (cd "$directory" && shasum -a 256 -c SHA256SUMS)
  fi
}

hash_check "$pointwise"
hash_check "$projection"
sh -n "$repo_root/scripts/probe-luna-bf16-families-cuda.sh"

for source in \
  "$pointwise/embedding.cu" "$pointwise/rms_norm.cu" \
  "$pointwise/residual.cu" "$pointwise/rope.cu" \
  "$projection/qkv.cu" "$projection/dense.cu" \
  "$projection/mlp.cu" "$projection/lm_head.cu"
do
  grep -F 'extern "C" __global__ void lunaflux_luna_' "$source" >/dev/null
  if grep -E 'cuda_runtime|cudaMalloc|cudaMemcpy|malloc|free|cublas|nvrtc|system\(' "$source" >/dev/null; then
    echo "forbidden runtime authority in $source" >&2
    exit 1
  fi
done

grep -F 'cuModuleLoad' "$repo_root/scripts/luna-bf16-family-driver-probe.cpp" >/dev/null
grep -F 'cuLaunchKernel' "$repo_root/scripts/luna-bf16-family-driver-probe.cpp" >/dev/null
if grep -E 'cuda_runtime|cudaMalloc|cudaMemcpy|cudaDeviceSynchronize' \
  "$repo_root/scripts/luna-bf16-family-driver-probe.cpp" >/dev/null
then
  echo "probe must use the CUDA Driver API only" >&2
  exit 1
fi

cd "$repo_root"
moon test --target native \
  kernels/luna_cuda_pointwise_aot/physical_fixture_test.mbt --deny-warn
moon test --target native \
  kernels/luna_cuda_projection_aot/physical_fixture_wbtest.mbt --deny-warn

echo "luna BF16 family physical fixture gate passed"
