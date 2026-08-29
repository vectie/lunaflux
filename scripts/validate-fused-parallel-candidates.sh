#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

package=kernels/luna_cuda_fused_parallel_aot
export_package=release/luna_fused_candidate_export
referee_package=tests/fused_parallel_qualification

for file in "$package"/*.mbt "$export_package"/*.mbt "$referee_package"/*.mbt; do
  lines=$(wc -l < "$file" | tr -d ' ')
  [ "$lines" -le 500 ] || {
    printf '%s\n' "fused candidate source exceeds 500 lines: $file" >&2
    exit 1
  }
done

grep -F 'QkvRopeKvWriteOrderedF32TranscendentalToleranceV1' \
  "$package/types.mbt" >/dev/null
grep -F 'ResidualRmsNormBlock128F32ReductionToleranceV1' \
  "$package/types.mbt" >/dev/null
grep -F 'blockDim.x != LF_BLOCK_THREADS' \
  "$package/source_qkv_rope_kv.mbt" >/dev/null
grep -F 'atomicAdd(dispatch_canary' \
  "$package/source_qkv_rope_kv.mbt" >/dev/null
grep -F 'decode != rows - prefill' \
  "$package/source_qkv_rope_kv.mbt" >/dev/null
grep -F 'query_row_offsets[rows] != tokens' \
  "$package/source_qkv_rope_kv.mbt" >/dev/null
grep -F 'page_table_offsets[rows] != page_cells' \
  "$package/source_qkv_rope_kv.mbt" >/dev/null
grep -F 'position >= sequence_lengths[row]' \
  "$package/source_qkv_rope_kv.mbt" >/dev/null
grep -F 'physical_page >= LF_TOTAL_PAGE_COUNT' \
  "$package/source_qkv_rope_kv.mbt" >/dev/null
grep -F 'projected = __bfloat162float(__float2bfloat16_rn(projected))' \
  "$package/source_qkv_rope_kv.mbt" >/dev/null
grep -F 'weight[weight_column * LF_INPUT_WIDTH + inner]' \
  "$package/source_qkv_rope_kv.mbt" >/dev/null
grep -F 'paired_weight[paired_weight_column * LF_INPUT_WIDTH + inner]' \
  "$package/source_qkv_rope_kv.mbt" >/dev/null
grep -F 'projected = __fadd_rn(projected, product)' \
  "$package/source_qkv_rope_kv.mbt" >/dev/null
grep -F '__shared__ float partial[LF_BLOCK_THREADS]' \
  "$package/source_residual_rmsnorm.mbt" >/dev/null
grep -F 'decode != rows - prefill' \
  "$package/source_residual_rmsnorm.mbt" >/dev/null
grep -F 'token >= tokens' \
  "$package/source_residual_rmsnorm.mbt" >/dev/null
grep -F 'const float rounded = __bfloat162float(encoded)' \
  "$package/source_residual_rmsnorm.mbt" >/dev/null
grep -F '__nv_bfloat16* residual_output, __nv_bfloat16* norm_output)' \
  "$package/source_residual_rmsnorm_production.mbt" >/dev/null
grep -F 'fused_qkv_rope_kv_source(symbol, parameters, profile, None)' \
  "$package/source_qkv_rope_kv_production.mbt" >/dev/null
if rg -F 'atomicAdd(' "$package/source_residual_rmsnorm_production.mbt" ||
  rg -F 'unsigned int* dispatch_canary' \
    "$package/source_residual_rmsnorm_production.mbt"; then
  printf '%s\n' 'production fused candidate retained diagnostic device work' >&2
  exit 1
fi
if rg -F 'atomicAdd(' "$package/source_qkv_rope_kv_production.mbt" ||
  rg -F 'Some(' "$package/source_qkv_rope_kv_production.mbt"; then
  printf '%s\n' 'production fused QKV source path enabled diagnostics' >&2
  exit 1
fi
grep -F 'diagnostic_canary=absent' "$package/recipe.mbt" >/dev/null
grep -F 'execution_policy=production-fast-path' "$package/recipe.mbt" >/dev/null
grep -F 'runtime_jit=false' "$package/recipe.mbt" >/dev/null
grep -F 'release_binding=candidate-only-promotion-required' \
  "$package/recipe.mbt" >/dev/null
grep -F 'performance_claim=none' "$package/promotion.mbt" >/dev/null
grep -F 'manifest_bindable=false' "$package/promotion.mbt" >/dev/null
grep -F 'measurement_authority=caller-supplied-raw-claim' \
  "$package/promotion_evidence.mbt" >/dev/null
grep -F 'physical_performance_authority=external' \
  "$package/promotion_evidence.mbt" >/dev/null
grep -F 'candidate_total >= baseline_total' \
  "$package/promotion_evidence.mbt" >/dev/null
grep -F 'FusedParallelMismatch(Correctness)' \
  "$package/promotion_evidence.mbt" >/dev/null
grep -F 'static_resource_audit_sha256=' \
  "$package/promotion_evidence.mbt" >/dev/null
grep -F 'schema=lunaflux-fused-parallel-compiled-candidate-binding.v1' \
  "$package/compiled_artifact.mbt" >/dev/null
grep -F 'expected_compile_receipt_digest' \
  "$package/compiled_artifact.mbt" >/dev/null
grep -F 'first_digest != second_digest || first_module != second_module' \
  "$package/compiled_artifact.mbt" >/dev/null
grep -F 'fused_require_recipe_abi(' \
  "$package/compiled_artifact.mbt" >/dev/null
grep -F 'compiler_authority=absent' \
  "$package/compiled_artifact.mbt" >/dev/null
grep -F 'deployment_approval=absent' \
  "$package/compiled_artifact.mbt" >/dev/null
grep -F 'promotion_authority=absent' \
  "$package/compiled_artifact.mbt" >/dev/null
grep -F 'manifest_bindable=false' \
  "$package/compiled_artifact.mbt" >/dev/null

for signature in \
  'pub fn bind_fused_qkv_rope_kv_compiled_candidate(FusedQkvRopeKvWriteCudaCandidate, Bytes, Bytes, Bytes, FusedParallelDigest, FusedParallelCompiledArtifactLimits) -> FusedParallelCompiledArtifactBinding raise FusedParallelAotError' \
  'pub fn bind_fused_residual_rmsnorm_compiled_candidate(FusedResidualRmsNormCudaCandidate, Bytes, Bytes, Bytes, FusedParallelDigest, FusedParallelCompiledArtifactLimits) -> FusedParallelCompiledArtifactBinding raise FusedParallelAotError' \
  'pub fn bind_fused_residual_rmsnorm_production_compiled_candidate(FusedResidualRmsNormProductionCudaCandidate, Bytes, Bytes, Bytes, FusedParallelDigest, FusedParallelCompiledArtifactLimits) -> FusedParallelCompiledArtifactBinding raise FusedParallelAotError'; do
  grep -F -x "$signature" "$package/pkg.generated.mbti" >/dev/null || {
    printf '%s\n' 'fused compiled-candidate binder signature drifted' >&2
    exit 1
  }
done

if rg -n \
  'KernelModuleInput|KernelEntryPointInput|LunaKernelCompiledOperation|LunaAotKernelAdmission|Approved|PromotionBinding' \
  "$package/compiled_artifact.mbt" \
  "$package/compiled_artifact_types.mbt" \
  "$package/compiled_artifact_accessors.mbt" >/dev/null; then
  printf '%s\n' 'fused compiled candidate gained runtime, approval, or promotion authority' >&2
  exit 1
fi

if rg -n 'nvrtc|cudaMalloc|cudaFree|malloc\(|calloc\(|realloc\(|cublas|#include <torch|python' \
  "$package/source_qkv_rope_kv.mbt" \
  "$package/source_residual_rmsnorm.mbt" \
  "$package/source_residual_rmsnorm_production.mbt" \
  "$package/promotion_evidence.mbt" >/dev/null; then
  printf '%s\n' 'fused candidate source gained runtime/compiler allocation authority' >&2
  exit 1
fi

if rg -n 'engine/device_step|config/runtime_descriptor|cmd/lunaflux' \
  "$package/moon.pkg" >/dev/null; then
  printf '%s\n' 'fused candidate package crossed into runtime wiring' >&2
  exit 1
fi

grep -F 'schema=lunaflux-fused-candidate-qualification-set.v1' \
  "$export_package/prepare.mbt" >/dev/null
grep -F 'target=sm_120' "$export_package/prepare.mbt" >/dev/null
grep -F 'block_threads=128' "$export_package/prepare.mbt" >/dev/null
grep -F 'qualification_only=1' "$export_package/prepare.mbt" >/dev/null
grep -F 'manifest_bindable=0' "$export_package/prepare.mbt" >/dev/null
grep -F 'compiler_authority=0' "$export_package/prepare.mbt" >/dev/null
grep -F 'artifact_authority=0' "$export_package/prepare.mbt" >/dev/null
grep -F 'device_authority=0' "$export_package/prepare.mbt" >/dev/null
grep -F 'runtime_authority=0' "$export_package/prepare.mbt" >/dev/null
grep -F 'promotion_authority=0' "$export_package/prepare.mbt" >/dev/null
grep -F 'replace=false' "$export_package/export.mbt" >/dev/null
grep -F 'CreateNew' "$export_package/export.mbt" >/dev/null

grep -F 'single-page-tail' "$referee_package/shapes.mbt" >/dev/null
grep -F 'cross-page-pair' "$referee_package/shapes.mbt" >/dev/null
grep -F 'eight-token-page-boundaries' "$referee_package/shapes.mbt" >/dev/null
grep -F 'FUSED_QUALIFICATION_BLOCK_THREADS : Int = 128' \
  "$referee_package/constants.mbt" >/dev/null
grep -F 'fused_qualification_qkv_reference' \
  "$referee_package/qkv_referee.mbt" >/dev/null
grep -F 'fused_qualification_residual_reference' \
  "$referee_package/residual_referee.mbt" >/dev/null

if rg -n 'internal/cuda|vectie/lunaflux/device|engine/device_step|kernels/artifact|kernels/luna_kernel_bundle|config/runtime_descriptor|cmd/lunaflux' \
  "$export_package/moon.pkg" "$referee_package/moon.pkg" >/dev/null; then
  printf '%s\n' 'fused qualification packages crossed into device/runtime/artifact wiring' >&2
  exit 1
fi

if rg -n 'nvrtc|cudaMalloc|cudaFree|cuModule|cublasLtCreate|cublasLtMatmul|nccl|fork\(|exec\(|system\(' \
  "$export_package" "$referee_package" >/dev/null; then
  printf '%s\n' 'fused qualification packages gained compiler/device/process authority' >&2
  exit 1
fi

moon fmt --check "$package"
moon fmt --check "$export_package" "$referee_package"
moon check "$package" --target native --deny-warn --warn-list +73
moon test "$package" --target native --deny-warn --warn-list +73
moon check "$export_package" "$referee_package" \
  --target native --deny-warn --warn-list +73
moon test "$export_package" "$referee_package" \
  --target native --deny-warn --warn-list +73

printf '%s\n' \
  'Fused candidates remain authority-free; production residual and QKV ABIs are canary-free.'
