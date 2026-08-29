#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

evidence=release/luna_fused_physical_evidence
package=kernels/luna_fused_artifact_admission

for source in "$evidence"/*.mbt "$package"/*.mbt; do
  lines=$(wc -l < "$source" | tr -d ' ')
  [ "$lines" -lt 500 ] || {
    printf 'approved fused manifest source exceeds file budget: %s (%s)\n' \
      "$source" "$lines" >&2
    exit 1
  }
done

if rg -n \
  'internal/(cuda|process|approved_fs)|moonbitlang/async|moonbitlang/x/sys|extern[[:space:]]+"[cC]"|@(fs|sys|process|cuda)\.' \
  "$package" --glob '*.mbt' --glob 'moon.pkg' >/dev/null; then
  printf '%s\n' 'approved fused artifact admission gained active authority' >&2
  exit 1
fi

for required in \
  'pub fn admit_approved_fused_production_manifest(' \
  'schema=lunaflux-fused-production-manifest.v1' \
  'pub fn admit_approved_fused_kernel_bundle(' \
  'pub fn admit_approved_fused_kernel_catalog(' \
  'pub fn adapt_approved_fused_runtime_identities(' \
  'pub fn ApprovedFusedKernelCatalog::resolve_exact(' \
  'standalone_fallback_required=true' \
  'runtime_dispatch_authority=approved-exact-manifest-only'; do
  rg -F -q "$required" "$evidence" "$package" --glob '*.mbt' || {
    printf 'approved fused artifact boundary missing: %s\n' "$required" >&2
    exit 1
  }
done

for required in \
  'record.residual_compiled.artifact_purpose() != ProductionFastPath' \
  'record.residual_compiled.kernel_abi() != ResidualRmsNormProductionAbiV2' \
  'record.residual_compiled.dispatch_canary_per_token() != 0U' \
  'pub fn ApprovedFusedKernelModule::artifact_purpose('; do
  rg -F -q "$required" "$evidence" "$package" --glob '*.mbt' || {
    printf 'canary-free production manifest boundary missing: %s\n' \
      "$required" >&2
    exit 1
  }
done

if rg -n \
  '^pub fn (admit_approved_fused_production_manifest|admit_approved_fused_kernel_bundle|admit_approved_fused_kernel_catalog)\([^A]*(@luna_cuda_fused_parallel_aot\.(FusedQkv|FusedResidual|FusedParallelCompiled)|AdmittedFusedPhysicalEvidence|FusedManifestApprovalRecord|FusedCandidateBenchmarkQualificationEvidence)' \
  "$evidence" "$package" --glob '*.mbt' >/dev/null; then
  printf '%s\n' 'unapproved fused fragment reached production admission' >&2
  exit 1
fi

if rg -n \
  '^pub fn (ApprovedFusedProductionManifest|ApprovedFusedKernelBundle|ApprovedFusedKernelCatalog)::(new|make|create|from_|open|load|execute)' \
  "$evidence" "$package" --glob '*.mbt' --glob 'pkg.generated.mbti' >/dev/null; then
  printf '%s\n' 'approved fused production authority became fabricable' >&2
  exit 1
fi

# The reviewed runtime consumers are the device-step projection and the worker
# plan that privately retains that inert startup projection, plus the isolated
# child bootstrap that consumes the descriptor-pinned source before roots close.
for route in cmd device runtime service; do
  [ ! -d "$route" ] ||
    if rg -n \
      'ApprovedFusedProductionManifest|ApprovedFusedKernel(Bundle|Catalog)|luna_fused_artifact_admission' \
      "$route" --glob '*.mbt' --glob 'moon.pkg' >/dev/null; then
      printf 'approved fused catalog crossed the reviewed worker boundary: %s\n' \
        "$route" >&2
      exit 1
    fi
done
if rg -n \
  'ApprovedFusedProductionManifest|ApprovedFusedKernel(Bundle|Catalog)|luna_fused_artifact_admission' \
  engine --glob '*.mbt' --glob 'moon.pkg' |
  grep -v '^engine/device_step/' |
  grep -v '^engine/device_worker/' |
  grep -v '^engine/device_worker_bootstrap/' >/dev/null; then
  printf '%s\n' 'approved fused catalog escaped reviewed runtime ownership' >&2
  exit 1
fi

if rg -n 'vectie/lunaflux/release/' engine/device_step engine/device_worker \
  --glob 'moon.pkg' >/dev/null; then
  printf '%s\n' 'fused runtime consumer imported release evidence' >&2
  exit 1
fi

moon fmt --check "$evidence" "$package"
moon check "$evidence" --target native --deny-warn --warn-list +73
moon check "$package" --target native --deny-warn --warn-list +73
moon test "$evidence" --target native --deny-warn --warn-list +73
moon test "$package" --target native --deny-warn --warn-list +73
moon info "$evidence" --target native >/dev/null
moon info "$package" --target native >/dev/null

evidence_interface="$evidence/pkg.generated.mbti"
package_interface="$package/pkg.generated.mbti"

grep -F -x \
  'pub fn admit_approved_fused_production_manifest(ApprovedFusedManifestArtifacts) -> ApprovedFusedProductionManifest raise FusedProductionManifestError' \
  "$evidence_interface" >/dev/null
grep -F -x \
  'pub fn admit_approved_fused_kernel_bundle(@luna_cuda_fused_parallel_aot.FusedParallelDigest, @luna_cuda_fused_parallel_aot.FusedParallelDigest, @catalog.DeviceTarget, @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding, @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding, @luna_cuda_fused_parallel_aot.FusedParallelDigest, @luna_cuda_fused_parallel_aot.FusedParallelDigest, UInt, UInt, ReadOnlyArray[FusedStandaloneFallbackIdentity], ReadOnlyArray[FusedStandaloneFallbackIdentity], @luna_capability_manifest.LunaAdmittedExternalRecord) -> ApprovedFusedKernelBundle raise FusedArtifactAdmissionError' \
  "$package_interface" >/dev/null
grep -F -x \
  'pub fn admit_approved_fused_kernel_catalog(@luna_cuda_fused_parallel_aot.FusedParallelDigest, @luna_cuda_fused_parallel_aot.FusedParallelDigest, @catalog.DeviceTarget, @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding, @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding, @luna_cuda_fused_parallel_aot.FusedParallelDigest, @luna_cuda_fused_parallel_aot.FusedParallelDigest, UInt, UInt, ReadOnlyArray[FusedStandaloneFallbackIdentity], ReadOnlyArray[FusedStandaloneFallbackIdentity], @luna_capability_manifest.LunaAdmittedExternalRecord) -> ApprovedFusedKernelCatalog raise FusedArtifactAdmissionError' \
  "$package_interface" >/dev/null
grep -F -x \
  'pub fn adapt_approved_fused_runtime_identities(ApprovedFusedProductionManifest) -> (@luna_fused_artifact_admission.ApprovedFusedKernelCatalog, @luna_fused_artifact_admission.ApprovedFusedKernelBundle) raise FusedProductionManifestError' \
  "$evidence_interface" >/dev/null
grep -F -x \
  'pub fn ApprovedFusedKernelCatalog::resolve_exact(Self, @luna_cuda_fused_parallel_aot.FusedParallelDigest, @catalog.DeviceTarget, @luna_cuda_fused_parallel_aot.FusedParallelFamily, @luna_cuda_fused_parallel_aot.FusedParallelDigest, @luna_cuda_fused_parallel_aot.FusedParallelDigest, @luna_cuda_fused_parallel_aot.FusedParallelDigest, String, String, String, @luna_cuda_fused_parallel_aot.FusedParallelKernelAbi, @launch_contract.AotLaunchDimensions, @luna_cuda_fused_parallel_aot.FusedParallelDigest, @luna_cuda_fused_parallel_aot.FusedParallelDigest) -> ApprovedFusedKernelModule raise FusedArtifactAdmissionError' \
  "$package_interface" >/dev/null || {
  printf '%s\n' 'approved fused catalog exact-selection interface drifted' >&2
  exit 1
}

if [ "$(rg -c '^pub fn admit_approved_fused_production_manifest\(' "$evidence_interface")" -ne 1 ] ||
  [ "$(rg -c '^pub fn admit_approved_fused_kernel_(bundle|catalog)\(' "$package_interface")" -ne 2 ]; then
  printf '%s\n' 'approved fused authority gained an alternate admission route' >&2
  exit 1
fi

for opaque in \
  ApprovedFusedProductionManifest \
  ApprovedFusedKernelBundle \
  ApprovedFusedKernelCatalog; do
  rg -U -q "pub struct $opaque \\{\\n  // private fields\\n\\}" \
    "$evidence_interface" "$package_interface" || {
    printf 'approved fused production type is no longer opaque: %s\n' \
      "$opaque" >&2
    exit 1
  }
done

printf '%s\n' 'approved fused manifest bundle/catalog admission gate passed'
