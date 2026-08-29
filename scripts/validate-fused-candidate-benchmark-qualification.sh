#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

package=benchmarks/luna_fused_candidate_qualification

for file in "$package"/*.mbt; do
  lines=$(wc -l < "$file" | tr -d ' ')
  [ "$lines" -le 500 ] || {
    printf '%s\n' "fused benchmark qualification source exceeds 500 lines: $file" >&2
    exit 1
  }
done

grep -F 'const FUSED_BENCHMARK_WARMUP_ITERATIONS : Int = 16' \
  "$package/protocol.mbt" >/dev/null
grep -F 'const FUSED_BENCHMARK_TRIALS_PER_SHAPE : Int = 8' \
  "$package/protocol.mbt" >/dev/null
grep -F 'const FUSED_BENCHMARK_REPETITIONS_PER_ARM : Int = 64' \
  "$package/protocol.mbt" >/dev/null
grep -F 'const FUSED_BENCHMARK_MINIMUM_WIN_PERCENT : Int64 = 5L' \
  "$package/protocol.mbt" >/dev/null
grep -F 'shape_count=4' "$package/shapes.mbt" >/dev/null
grep -F 'candidate.duration_ns >= observation.baseline.duration_ns' \
  "$package/validate.mbt" >/dev/null
grep -F 'baseline.total * (100L - FUSED_BENCHMARK_MINIMUM_WIN_PERCENT)' \
  "$package/validate.mbt" >/dev/null
grep -F 'observation.resources_acquired != observation.resources_released' \
  "$package/validate.mbt" >/dev/null
grep -F 'observation.candidate.dispatch_canary_after -' \
  "$package/validate.mbt" >/dev/null
grep -F 'manifest_bindable=false' "$package/canonical.mbt" >/dev/null
grep -F 'promotion_authority=false' "$package/canonical.mbt" >/dev/null
grep -F 'measurement_authority=caller-supplied-raw-observations' \
  "$package/canonical.mbt" >/dev/null

for signature in \
  'pub fn qualify_fused_qkv_candidate_benchmark(@luna_cuda_fused_parallel_aot.FusedQkvRopeKvWriteCudaCandidate, FusedCandidateBenchmarkProtocol, ArrayView[FusedCandidateBenchmarkPairedObservation], maximum_evidence_bytes~ : Int) -> FusedCandidateBenchmarkQualificationEvidence raise FusedCandidateBenchmarkError' \
  'pub fn qualify_fused_residual_rmsnorm_candidate_benchmark(@luna_cuda_fused_parallel_aot.FusedResidualRmsNormCudaCandidate, FusedCandidateBenchmarkProtocol, ArrayView[FusedCandidateBenchmarkPairedObservation], maximum_evidence_bytes~ : Int) -> FusedCandidateBenchmarkQualificationEvidence raise FusedCandidateBenchmarkError'; do
  grep -F -x "$signature" "$package/pkg.generated.mbti" >/dev/null || {
    printf '%s\n' 'fused benchmark qualification API drifted' >&2
    exit 1
  }
done

if rg -n \
  'benchmarks/evidence|benchmarks/runner|internal/cuda|vectie/lunaflux/device|engine/device_step|kernels/luna_kernel_bundle|config/runtime_descriptor|cmd/lunaflux|FusedParallelPromotionBinding|bind_fused_.*promotion' \
  "$package/moon.pkg" "$package"/*.mbt >/dev/null; then
  printf '%s\n' 'fused benchmark qualification crossed into runner, device, runtime, or promotion authority' >&2
  exit 1
fi

if rg -n \
  'cudaMalloc|cudaFree|cuModule|nvrtc|cublasLt|nccl|fork\(|exec\(|system\(|clock_gettime|mach_absolute_time' \
  "$package"/*.mbt >/dev/null; then
  printf '%s\n' 'fused benchmark qualification gained measurement or execution authority' >&2
  exit 1
fi

moon fmt --check "$package"
moon check "$package" --target native --deny-warn --warn-list +73
moon test "$package" --target native --deny-warn --warn-list +73

printf '%s\n' 'Fused candidate benchmark evidence remains fixed-shape and qualification-only.'
