#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

fail() {
  printf '%s\n' "Luna BF16 kernel promotion boundary: $1" >&2
  exit 1
}

package=kernels/luna_bf16_kernel_producer
interface="$package/pkg.generated.mbti"

for source in "$package"/promotion_*.mbt; do
  lines=$(wc -l < "$source" | tr -d ' ')
  [ "$lines" -lt 500 ] || fail "$source exceeds the 499-line source budget"
done

if rg -n 'try!|open_context|load_module|load_function|nvrtc|std/process|async/fs' \
  "$package/promotion_types.mbt" \
  "$package/promotion_collect.mbt" \
  "$package/promotion_canonical.mbt" \
  "$package/promotion_join.mbt"; then
  fail 'production promotion path acquired execution/compiler/process authority'
fi

for surface in \
  'pub fn collect_luna_bf16_profiler_evidence(LunaBf16CandidateSet, @luna_profile_priority.LunaPagedKernelProfileCapture, ArrayView[LunaBf16MicrobenchmarkObservation], ArrayView[LunaBf16EndToEndObservation], LunaBf16ProfilerEvidenceLimits) -> LunaBf16ProfilerEvidence raise LunaBf16PromotionError' \
  'pub fn promote_luna_bf16_kernel_release(LunaBf16CandidateSet, LunaBf16KernelReleaseInput, @luna_profile_priority.LunaPagedKernelProfileCapture, LunaBf16ProfilerEvidence, LunaBf16PromotionLimits) -> LunaBf16OfflinePromotion raise LunaBf16PromotionError'; do
  rg -F -q "$surface" "$interface" || fail "public surface drifted: $surface"
done

for invariant in \
  'candidate_total >= baseline_total' \
  'LunaBf16PromotionRejected(EndToEndRegression)' \
  'operation.source_digest_sha256() != candidate.source_sha256()' \
  'operation.recipe_digest_sha256() != candidate.recipe_sha256()' \
  'operation.toolchain_digest_sha256()' \
  'operation.driver_identity_sha256() != driver_identity' \
  'encode_launch_contracts(' \
  'bundle.execution_manifest().operation_count()'; do
  rg -F -q "$invariant" "$package"/promotion_*.mbt || \
    fail "missing promotion invariant: $invariant"
done

moon info >/dev/null
moon check --target native --deny-warn --warn-list +73 "$package"
moon test --target native --deny-warn --warn-list +73 "$package"

printf '%s\n' 'Luna BF16 kernel promotion boundary validation passed.'
