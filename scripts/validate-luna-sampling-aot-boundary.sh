#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
package_dir="$repo_root/kernels/luna_cuda_sampling_aot"

test -f "$package_dir/moon.pkg"

if rg -n 'internal/cuda|core/(fs|process|stdio)|moonbitlang/x/(fs|process)|device/' \
  "$package_dir/moon.pkg" "$package_dir"/*.mbt; then
  echo "sampling boundary imported runtime, device, filesystem, or process authority" >&2
  exit 1
fi

if rg -n 'nvrtc|cuModule|cuLaunchKernel|internal/cuda' "$package_dir"/*.mbt; then
  echo "sampling AOT package acquired runtime compilation or CUDA authority" >&2
  exit 1
fi

operation_vocabulary=$(sed -n '/pub(all) enum OperationKind {/,/}/p' \
  "$repo_root/model/plan/operations.mbt")
if printf '%s\n' "$operation_vocabulary" | rg -qi 'sampl|logit.*reduc'; then
  echo "model graph unexpectedly gained a sampling operation" >&2
  exit 1
fi

rg -Fq 'LanguageModelHead,' "$repo_root/model/llama/builder.mbt"
rg -Fq 'lunaflux_greedy_bf16_rows_v1' "$package_dir/greedy_lower.mbt"
rg -Fq '!isfinite(value)' "$package_dir/greedy_lower.mbt"
rg -Fq 'value == best && token < best_token' "$package_dir/greedy_lower.mbt"
rg -Fq 'first_module != second_module' "$package_dir/greedy_artifact.mbt"
rg -Fq 'PagedCudaGreedySampling' \
  "$repo_root/engine/device_step/paged_executor_types.mbt"
rg -Fq 'stage_frame_with_sampling_mode' \
  "$repo_root/engine/device_step/paged_executor_run.mbt"
rg -Fq 'stage_with_sampling_mode' \
  "$repo_root/engine/device_step/paged_executor_run.mbt"
rg -Fq 'sampling_mode_value()' \
  "$repo_root/engine/device_step/preflight_frame.mbt"
rg -Fq 'copy_paged_cuda_greedy_results' \
  "$repo_root/engine/device_step/paged_executor_completion.mbt"
rg -Fq 'sampling_policy~' "$repo_root/engine/device_worker/prepare.mbt"
rg -Fq 'module_.module_bytes() == artifact.module_bytes()' \
  "$repo_root/engine/device_step/paged_executor_greedy_sampling.mbt"
rg -Fq 'PAGED_GREEDY_RESULT_ROW_BYTES : Int64 = 8L' \
  "$repo_root/engine/device_step/paged_executor_greedy_sampling.mbt"
rg -Fq '@sampling.greedy(' \
  "$repo_root/engine/device_step/paged_executor_completion.mbt"
rg -Fq '@sampling.stochastic_sample_at(' \
  "$repo_root/engine/device_step/paged_executor_completion.mbt"
if rg -n 'nvrtc|PyTorch|python|runtime JIT' \
  "$package_dir"/*.mbt \
  "$repo_root/engine/device_step/paged_executor_greedy_sampling.mbt"; then
  echo "sampling fast path introduced a forbidden runtime dependency" >&2
  exit 1
fi

for source in "$package_dir"/*.mbt; do
  lines=$(wc -l < "$source" | tr -d ' ')
  if [ "$lines" -gt 500 ]; then
    echo "sampling-boundary source exceeds 500 lines: $source ($lines)" >&2
    exit 1
  fi
done

cd "$repo_root"
moon check --target native --deny-warn --warn-list +73 kernels/luna_cuda_sampling_aot
moon test --target native --deny-warn --warn-list +73 kernels/luna_cuda_sampling_aot
moon check --target native --deny-warn --warn-list +73 engine/device_step
moon test --target native --deny-warn --warn-list +73 engine/device_step

echo "sampling-reduction ownership boundary: pass (authenticated greedy AOT; explicit host stochastic mode)"
