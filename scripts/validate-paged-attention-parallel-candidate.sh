#!/bin/sh

set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

package=kernels/luna_cuda_paged_attention_parallel_aot

require() {
  pattern=$1
  shift
  if ! rg -q "$pattern" "$@"; then
    printf '%s\n' "missing parallel paged-attention boundary: $pattern" >&2
    exit 1
  fi
}

require 'block_x=PAGED_ATTENTION_PARALLEL_BLOCK_THREADS' "$package/lower.mbt"
require 'PAGED_ATTENTION_PARALLEL_BLOCK_THREADS : Int = 128' "$package/types.mbt"
require 'extern __shared__ float shared\[\]' "$package/source.mbt"
require 'lf_parallel_reduce_sum_v1' "$package/source.mbt"
require 'lf_parallel_reduce_max_v1' "$package/source.mbt"
require 'linear \+= LF_BLOCK_THREADS' "$package/source.mbt"
require 'plan_luna_tile_cuda_aot_input' "$package/tile_plan.mbt"
require 'target.compute_major\(\) < 8' "$package/lower.mbt"
require 'head_dimension % 16 != 0' "$package/lower.mbt"
require 'tokens_per_page < 8' "$package/lower.mbt"
require 'maximum_context_tokens > PAGED_ATTENTION_PARALLEL_MAX_CONTEXT' "$package/lower.mbt"
require 'bf16-input-fp32-block128-tree-softmax-tolerance-v1' "$package/types.mbt"
require 'reference_fallback_source_sha256' "$package/lower.mbt"
require 'release_binding=candidate-only-promotion-required' "$package/lower.mbt"
require 'output=cubin' "$package/lower.mbt"
require 'language_standard=c\+\+17' "$package/lower.mbt"
require 'admit_paged_attention_parallel_promotion_evidence' "$package/promotion.mbt"
require 'candidate_total >= baseline_total' "$package/promotion.mbt"
require 'manifest_bindable=false' "$package/lower.mbt" "$package/promotion.mbt"

if rg -q 'block=1,1,1|block_x=1|RuntimeInstance|online_tcp|device_worker|nvrtc|JIT' \
  "$package" --glob '*.mbt'; then
  printf '%s\n' 'parallel candidate regained serial geometry or runtime authority' >&2
  exit 1
fi

if rg -q 'manifest_bindable=true|performance_win=true|benchmark_passed=true' \
  "$package" --glob '*.mbt' --glob '*.md'; then
  printf '%s\n' 'parallel candidate overclaims release or performance authority' >&2
  exit 1
fi

printf '%s\n' 'Parallel paged-attention candidate boundaries are valid.'
