#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

package=engine/device_step

for source in "$package"/fused_grouped_*.mbt; do
  lines=$(wc -l < "$source" | tr -d ' ')
  [ "$lines" -lt 500 ] || {
    printf 'fused grouped runtime source exceeds file budget: %s (%s)\n' \
      "$source" "$lines" >&2
    exit 1
  }
done

for required in \
  'pub fn admit_approved_fused_residual_rmsnorm_span(' \
  'pub fn admit_production_fused_residual_rmsnorm_span(' \
  'pub fn require_fused_qkv_read_only_attention_split(' \
  'QkvRequiresReadOnlyAttentionSplit' \
  'fused_dispatch_policy_allows_graph' \
  'prepare_fused_residual_resources' \
  'replace_paged_production_spans' \
  '!fused_dispatch_policy_allows_span(diagnostic_policy, span.kernel_abi())' \
  'ResidualRmsNormProductionAbiV2' \
  'fused_residual_argument_count_for_abi' \
  'copy_from_fixed_host' \
  'copy_to_fixed_host' \
  'fused_expected_dispatch_canary' \
  'dispatch_canary_per_token'; do
  rg -F -q "$required" "$package" \
    kernels/luna_cuda_fused_parallel_aot \
    kernels/luna_fused_artifact_admission \
    release/luna_fused_physical_evidence --glob '*.mbt' || {
    printf 'fused grouped runtime boundary missing: %s\n' "$required" >&2
    exit 1
  }
done


for production_boundary in \
  'test "production execution policy allocates no diagnostic host state"' \
  'test "fused execution policy separates qualification and production ABIs"' \
  'test "production residual RMSNorm ABI omits diagnostic argument"'; do
  rg -F -q "$production_boundary" "$package" --glob '*_wbtest.mbt' || {
    printf 'fused grouped runtime production boundary missing: %s\n' \
      "$production_boundary" >&2
    exit 1
  }
done

for graph_policy in \
  PagedOrderedEagerOnly \
  PagedCapturedRequired \
  PagedCapturedWithEagerFallback; do
  rg -F -q "$graph_policy" "$package/execution_diagnostics_wbtest.mbt" || {
    printf 'production fused graph policy coverage missing: %s\n' \
      "$graph_policy" >&2
    exit 1
  }
done

production_source=kernels/luna_cuda_fused_parallel_aot/source_residual_rmsnorm_production.mbt
for required in \
  'fused_residual_rmsnorm_production_source' \
  '__nv_bfloat16* residual_output, __nv_bfloat16* norm_output)'; do
  rg -F -q "$required" "$production_source" || {
    printf 'canary-free production source missing: %s\n' "$required" >&2
    exit 1
  }
done
if rg -F -q 'atomicAdd(' "$production_source" ||
  rg -F -q 'unsigned int* dispatch_canary' "$production_source"; then
  printf '%s\n' 'production fused source retained diagnostic device work' >&2
  exit 1
fi

for runtime_wiring in \
  'fused_residual_span? : ApprovedFusedResidualRmsNormSpan? = None' \
  'prepare_fused_residual_resources(context, span, resources)' \
  'replace_paged_production_spans('; do
  rg -F -q "$runtime_wiring" "$package" --glob '*.mbt' || {
    printf 'numeric BF16 fused runtime wiring missing: %s\n' \
      "$runtime_wiring" >&2
    exit 1
  }
done
rg -F -q 'fused_residual_span=plan.fused_residual_span' \
  engine/device_worker/prepare.mbt || {
  printf '%s\n' 'numeric BF16 worker did not retain admitted fused span' >&2
  exit 1
}
if rg -n 'copy_(from|to)_fixed_host|dispatch_canary' \
  "$package/numeric_bf16_executor.mbt" >/dev/null; then
  printf '%s\n' 'numeric BF16 production wiring gained diagnostic traffic' >&2
  exit 1
fi

if rg -n 'scheduler|service/|model/(llama|mistral)|runtime JIT|nvrtc' \
  "$package"/fused_grouped_*.mbt >/dev/null; then
  printf '%s\n' 'fused grouped runtime crossed scheduler/family/JIT boundary' >&2
  exit 1
fi

if rg -n '^pub fn ApprovedFusedResidualRmsNormSpan::(new|make|create|from_)' \
  "$package" --glob '*.mbt' --glob 'pkg.generated.mbti' >/dev/null; then
  printf '%s\n' 'fused grouped runtime authority became fabricable' >&2
  exit 1
fi

moon fmt --check "$package"/fused_grouped_*.mbt \
  "$package"/execution_diagnostics*.mbt \
  "$package/paged_executor_prepare.mbt" \
  "$package/paged_executor_run.mbt" \
  kernels/luna_cuda_fused_parallel_aot \
  kernels/luna_fused_artifact_admission release/luna_fused_physical_evidence
moon check "$package" --target native --deny-warn --warn-list +73
moon test "$package" --target native --deny-warn --warn-list +73
moon info "$package" --target native >/dev/null

interface="$package/pkg.generated.mbti"
grep -F 'pub fn admit_approved_fused_residual_rmsnorm_span(' \
  "$interface" >/dev/null
grep -F 'pub fn admit_production_fused_residual_rmsnorm_span(' \
  "$interface" >/dev/null
grep -F 'pub fn require_fused_qkv_read_only_attention_split(' \
  "$interface" >/dev/null
rg -U -q \
  'pub struct ApprovedFusedResidualRmsNormSpan \{\n  // private fields\n\}' \
  "$interface"

printf '%s\n' 'approved fused grouped-operation runtime gate passed'
