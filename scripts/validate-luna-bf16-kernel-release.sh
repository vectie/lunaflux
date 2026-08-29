#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
producer=$repo_root/kernels/luna_bf16_kernel_producer
root_plan=$repo_root/release/kernel_root

if rg -n \
  'internal/cuda|moonbitlang/async|moonbitlang/x/sys|extern[[:space:]]+"[cC]"|@(fs|sys)\.' \
  "$producer" "$root_plan" --glob '*.mbt' --glob 'moon.pkg'; then
  printf '%s\n' 'BF16 kernel release packages crossed their inert offline boundary' >&2
  exit 1
fi

if rg -n \
  'nvcc|ptxas|nvrtc|cuModuleLoadData|popen[[:space:]]*\(|system[[:space:]]*\(' \
  "$producer" "$root_plan" --glob '*.mbt' --glob 'moon.pkg'; then
  printf '%s\n' 'BF16 kernel release packages contain compiler, process, or loader authority' >&2
  exit 1
fi

find "$producer" "$root_plan" -type f -name '*.mbt' -print |
  while IFS= read -r source; do
    lines=$(wc -l <"$source" | tr -d ' ')
    if [ "$lines" -ge 500 ]; then
      printf '%s\n' "BF16 kernel release source exceeds 499 lines: $source ($lines)" >&2
      exit 1
    fi
  done

for lowerer in \
  "$repo_root/kernels/luna_cuda_pointwise_aot/lower.mbt" \
  "$repo_root/kernels/luna_cuda_projection_aot/lower.mbt" \
  "$repo_root/kernels/luna_cuda_paged_attention_aot/lower.mbt"; do
  rg -q 'pub fn lower_reusable_.*_cuda_aot_candidate' "$lowerer" || {
    printf '%s\n' "reusable release lowerer is absent: $lowerer" >&2
    exit 1
  }
done

for script in \
  luna-bf16-kernel-producer-common.sh \
  build-luna-bf16-kernel-set.sh \
  verify-luna-bf16-kernel-set.sh \
  test-luna-bf16-kernel-producer.sh \
  assemble-luna-kernel-root.sh \
  verify-luna-kernel-root.sh \
  test-luna-kernel-root-assembly.sh; do
  sh -n "$repo_root/scripts/$script"
done

cd "$repo_root"
moon fmt --check \
  kernels/luna_cuda_pointwise_aot \
  kernels/luna_cuda_projection_aot \
  kernels/luna_cuda_paged_attention_aot \
  kernels/luna_bf16_kernel_producer \
  release/kernel_root
moon check \
  kernels/luna_cuda_pointwise_aot \
  kernels/luna_cuda_projection_aot \
  kernels/luna_cuda_paged_attention_aot \
  kernels/luna_bf16_kernel_producer \
  release/kernel_root \
  --target native --deny-warn
moon test kernels/luna_bf16_kernel_producer --target native --deny-warn
"$repo_root/scripts/test-luna-bf16-kernel-producer.sh"
"$repo_root/scripts/test-luna-kernel-root-assembly.sh"

printf '%s\n' 'Luna BF16 kernel release producer gate passed'
