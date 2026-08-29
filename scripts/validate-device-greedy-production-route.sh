#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
sampling="$repo_root/kernels/luna_cuda_sampling_aot"
projection="$repo_root/kernels/luna_cuda_projection_aot"
step="$repo_root/engine/device_step"
worker="$repo_root/engine/device_worker"

rg -Fq 'lunaflux_greedy_bf16_rows_v1' "$sampling/greedy_lower.mbt"
rg -Fq 'embedded_greedy_sampling_cuda_source' "$projection/source_mlp.mbt"
rg -Fq 'parameters.family == LanguageModelHead' "$projection/lower.mbt"
rg -Fq 'embedded_sampling=greedy_bf16_rows_v1' "$projection/recipe.mbt"
rg -Fq 'admit_paged_embedded_cuda_greedy_sampling' \
  "$step/paged_executor_greedy_sampling.mbt"
rg -Fq 'stage_frame_with_sampling_mode' \
  "$step/paged_executor_run.mbt"
rg -Fq 'stage_with_sampling_mode' "$step/paged_executor_run.mbt"
rg -Fq 'sampling_mode_value()' "$step/preflight_frame.mbt"
rg -Fq 'copy_paged_cuda_greedy_results' \
  "$step/paged_executor_completion.mbt"
rg -Fq 'priv sampling_policy : @device_step.PagedSamplingExecutionPolicy' \
  "$worker/types.mbt"
[ "$(rg -c 'sampling_policy\? : @device_step.PagedSamplingExecutionPolicy = PagedHostSampling' "$worker/prepare.mbt")" -eq 2 ]
[ "$(rg -c 'sampling_policy=plan.sampling_policy' "$worker/prepare.mbt")" -eq 2 ]

if rg -n '@crypto|@approved_fs|@fs\.|@release|read_file|sha256\(' \
  "$step/paged_executor_greedy_sampling.mbt"; then
  echo 'device-greedy runtime path performs startup evidence or filesystem work' >&2
  exit 1
fi

cd "$repo_root"
moon check --target native --deny-warn --warn-list +73 \
  kernels/luna_cuda_sampling_aot kernels/luna_cuda_projection_aot \
  kernels/luna_bf16_kernel_producer engine/device_step engine/device_worker
moon test --target native --deny-warn --warn-list +73 \
  kernels/luna_cuda_sampling_aot kernels/luna_cuda_projection_aot \
  kernels/luna_bf16_kernel_producer engine/device_step engine/device_worker

echo 'device-greedy production route: pass (manifest-owned LM-head module; explicit host stochastic mode; no device fallback)'
