#!/bin/sh

set -eu
LC_ALL=C
export LC_ALL

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
cd "$repo_root"

package=release/luna_fused_physical_evidence

for file in "$package"/*.mbt; do
  lines=$(wc -l < "$file" | tr -d ' ')
  [ "$lines" -le 500 ] || {
    printf 'fused physical evidence file exceeds 500 lines: %s (%s)\n' \
      "$file" "$lines" >&2
    exit 1
  }
done

for dependency in \
  'vectie/lunaflux/benchmarks/luna_fused_candidate_qualification' \
  'vectie/lunaflux/internal/canonical_sha256' \
  'vectie/lunaflux/kernels/catalog' \
  'vectie/lunaflux/kernels/launch_contract' \
  'vectie/lunaflux/kernels/luna_capability_manifest' \
  'vectie/lunaflux/kernels/luna_cuda_fused_parallel_aot' \
  'vectie/lunaflux/kernels/luna_specialization'; do
  grep -F "\"$dependency\"" "$package/moon.pkg" >/dev/null || {
    printf 'fused physical evidence lost required typed dependency: %s\n' \
      "$dependency" >&2
    exit 1
  }
done

if rg -n \
  'moonbitlang/async|internal/(cuda|process|approved_fs)|vectie/lunaflux/(device|engine|runtime|service|cmd)/' \
  "$package/moon.pkg" >/dev/null; then
  printf '%s\n' \
    'fused physical evidence gained filesystem/device/process/runtime authority' >&2
  exit 1
fi

if rg -n \
  'extern[[:space:]]+"c"|@fs\.|@process\.|@cuda\.|async[[:space:]]+fn|pub fn (open|spawn|compile|execute|load_module|bind_manifest|promote)\(|pub fn [A-Za-z0-9_]+::(open|spawn|compile|execute|load_module|bind_manifest|promote)\(' \
  "$package" --glob '*.mbt' --glob '!**/*_test.mbt' >/dev/null; then
  printf '%s\n' \
    'fused physical evidence source gained active authority' >&2
  exit 1
fi

grep -F 'FUSED_PHYSICAL_CANONICAL_LINE_COUNT : Int = 172' \
  "$package/types.mbt" >/dev/null
grep -F 'FUSED_PHYSICAL_MAXIMUM_CANONICAL_BYTES : Int = 131072' \
  "$package/types.mbt" >/dev/null
grep -F 'lunaflux-fused-physical-qualification-evidence.v1' \
  "$package/admit.mbt" >/dev/null
grep -F 'lunaflux-fused-scalar-referee-shape-set.v1' \
  "$package/digest.mbt" >/dev/null
grep -F 'reader.finish()' "$package/admit.mbt" >/dev/null
grep -F 'first_build_cubin_sha256' "$package/admit_family.mbt" >/dev/null
grep -F 'second_build_cubin_sha256' "$package/admit_family.mbt" >/dev/null
grep -F 'compile_receipt_sha256' "$package/admit_family.mbt" >/dev/null
grep -F 'observed_absolute > maximum_absolute' \
  "$package/admit_family.mbt" >/dev/null
grep -F 'artifact_directory_sealed", "true"' "$package/admit.mbt" >/dev/null
grep -F 'evidence_directory_sealed", "true"' "$package/admit.mbt" >/dev/null
grep -F 'promotion_authority", "absent"' "$package/admit.mbt" >/dev/null
grep -F 'schema=lunaflux-fused-manifest-approval-record.v1' \
  "$package/promotion_join.mbt" >/dev/null
grep -F 'external_signature_required=true' \
  "$package/promotion_join.mbt" >/dev/null
grep -F '@luna_capability_manifest.admit_luna_external_approved_record' \
  "$package/promotion_join.mbt" >/dev/null
grep -F 'pub fn FusedManifestApprovalRecord::manifest_bindable' \
  "$package/promotion_accessors.mbt" >/dev/null
grep -F 'pub fn ApprovedFusedManifestArtifacts::manifest_bindable' \
  "$package/promotion_accessors.mbt" >/dev/null
grep -F 'does not perform' \
  "$package/README.mbt.md" >/dev/null
grep -F 'has exactly one' "$package/README.mbt.md" >/dev/null
grep -F 'existing qualification signature cannot be replayed as' \
  "$package/README.mbt.md" >/dev/null
grep -F 'pub struct LunaAuthenticatedExternalApproval {' \
  'kernels/luna_capability_manifest/signature.mbt' >/dev/null
if rg -n 'LunaExternalSignatureApprovalReceipt|from_deployment_approval' \
  kernels/luna_capability_manifest "$package" --glob '*.mbt'; then
  printf '%s\n' 'forgeable external approval construction remains' >&2
  exit 1
fi

# Only the separately reviewed immutable bundle/catalog seam may consume the
# approved wrapper. The reviewed device-step seam may consume only the opaque
# catalog fallback identities; all other production routes remain forbidden.
for route in \
  cmd \
  device \
  runtime \
  service \
  kernels/full_graph_manifest \
  kernels/luna_artifact_admission \
  kernels/luna_kernel_bundle; do
  [ ! -d "$route" ] ||
    if rg -n \
      'ApprovedFusedManifestArtifacts|admit_external_fused_manifest_approval|vectie/lunaflux/release/luna_fused_physical_evidence' \
      "$route" --glob '*.mbt' --glob 'moon.pkg' \
      --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt' >/dev/null; then
      printf 'production route gained unreviewed fused manifest authority: %s\n' \
        "$route" >&2
      exit 1
    fi
done
if rg -n \
  'ApprovedFusedManifestArtifacts|admit_external_fused_manifest_approval|vectie/lunaflux/release/luna_fused_physical_evidence' \
  engine --glob '*.mbt' --glob 'moon.pkg' \
  --glob '!**/*_test.mbt' --glob '!**/*_wbtest.mbt' |
  grep -v '^engine/device_step/' >/dev/null; then
  printf '%s\n' 'fused physical authority escaped device-step ownership' >&2
  exit 1
fi

moon fmt --check "$package"
moon check "$package" --target native --deny-warn --warn-list +73
moon test "$package" --target native --deny-warn --warn-list +73
moon info "$package" --target native >/dev/null

interface="$package/pkg.generated.mbti"
grep -F -x \
  'pub fn admit_fused_physical_evidence(@luna_cuda_fused_parallel_aot.FusedQkvRopeKvWriteCudaCandidate, @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding, @luna_cuda_fused_parallel_aot.FusedResidualRmsNormCudaCandidate, @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding, Bytes, expected_digest~ : @luna_cuda_fused_parallel_aot.FusedParallelDigest, expected_approved_policy_digest~ : @luna_cuda_fused_parallel_aot.FusedParallelDigest) -> AdmittedFusedPhysicalEvidence raise FusedPhysicalEvidenceError' \
  "$interface" >/dev/null || {
  printf '%s\n' 'fused physical evidence admission signature drifted' >&2
  exit 1
}
rg -U -q \
  'pub struct AdmittedFusedPhysicalEvidence \{\n  // private fields\n\}' \
  "$interface" || {
  printf '%s\n' 'fused physical evidence result is no longer opaque' >&2
  exit 1
}
if rg -n \
  '^pub fn AdmittedFusedPhysicalEvidence::(new|make|create|open|load|execute|promote)' \
  "$interface" >/dev/null; then
  printf '%s\n' 'fused physical evidence gained a constructor or authority method' >&2
  exit 1
fi

grep -F -x \
  'pub fn admit_external_fused_manifest_approval(FusedManifestApprovalRecord, @luna_capability_manifest.LunaDetachedSignatureEnvelope, @luna_capability_manifest.LunaAuthenticatedExternalApproval) -> ApprovedFusedManifestArtifacts raise FusedPhysicalEvidenceError' \
  "$interface" >/dev/null || {
  printf '%s\n' 'external fused manifest approval signature drifted' >&2
  exit 1
}
grep -F -x \
  'pub fn prepare_fused_manifest_approval_record(AdmittedFusedPhysicalEvidence, @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding, @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding, @luna_fused_candidate_qualification.FusedCandidateBenchmarkQualificationEvidence, @luna_fused_candidate_qualification.FusedCandidateBenchmarkQualificationEvidence) -> FusedManifestApprovalRecord raise FusedPhysicalEvidenceError' \
  "$interface" >/dev/null || {
  printf '%s\n' 'fused manifest approval-record signature drifted' >&2
  exit 1
}

printf '%s\n' \
  'Fused evidence join has one reviewed manifest projection and one fail-closed device-step consumer.'
