///|
/// Bounded reasons why an externally approved fused pair could not be
/// projected into the production manifest boundary.
pub(all) enum FusedProductionManifestRule {
  Authority
  Approval
  PhysicalEvidence
  BenchmarkEvidence
  Family
  Candidate
  Compiler
  Target
  Launch
  Symbol
  KernelAbi
  Artifact
  Fallback
  CanonicalDocument
} derive(Debug, Eq)

///|
pub(all) suberror FusedProductionManifestError {
  InvalidFusedProductionManifest(FusedProductionManifestRule)
  MismatchedFusedProductionManifest(FusedProductionManifestRule)
} derive(Debug, Eq)

///|

///|
/// One family entry inside an approved production pair. Construction remains
/// private so an inert compiled candidate cannot acquire manifest authority.
pub struct ApprovedFusedProductionFamily {
  priv compiled : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding
  priv benchmark : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv dispatch_canary_per_token : UInt
  priv fallbacks : ReadOnlyArray[
    @luna_fused_artifact_admission.FusedStandaloneFallbackIdentity,
  ]
}

///|
pub fn ApprovedFusedProductionFamily::family(
  self : ApprovedFusedProductionFamily,
) -> @luna_cuda_fused_parallel_aot.FusedParallelFamily {
  self.compiled.family()
}

///|
pub fn ApprovedFusedProductionFamily::compiled_artifact(
  self : ApprovedFusedProductionFamily,
) -> @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding {
  self.compiled
}

///|
pub fn ApprovedFusedProductionFamily::benchmark_digest(
  self : ApprovedFusedProductionFamily,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.benchmark
}

///|
pub fn ApprovedFusedProductionFamily::dispatch_canary_per_token(
  self : ApprovedFusedProductionFamily,
) -> UInt {
  self.dispatch_canary_per_token
}

///|
pub fn ApprovedFusedProductionFamily::fallbacks(
  self : ApprovedFusedProductionFamily,
) -> ReadOnlyArray[
  @luna_fused_artifact_admission.FusedStandaloneFallbackIdentity,
] {
  self.fallbacks
}

///|
/// Opaque production-manifest projection. Its constructor additionally
/// requires the canary-free residual/RMSNorm ABI; the current qualification
/// approval record therefore fails closed until a successor physical record
/// binds that exact production artifact.
pub struct ApprovedFusedProductionManifest {
  priv qkv : ApprovedFusedProductionFamily
  priv residual : ApprovedFusedProductionFamily
  priv target : @catalog.DeviceTarget
  priv physical : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv physical_artifact_seal : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv physical_evidence_seal : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv approval_record : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv approved_source : @luna_capability_manifest.LunaApprovedImmutableSourceIdentity
  priv approved_key : @luna_capability_manifest.LunaSignatureKeyIdentityDigest
  priv detached_signature : @luna_capability_manifest.LunaDetachedSignatureDigest
  priv approval : @luna_capability_manifest.LunaAdmittedExternalRecord
  priv canonical : Bytes
  priv digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
}

///|
pub fn ApprovedFusedProductionManifest::qkv_family(
  self : ApprovedFusedProductionManifest,
) -> ApprovedFusedProductionFamily {
  self.qkv
}

///|
pub fn ApprovedFusedProductionManifest::residual_family(
  self : ApprovedFusedProductionManifest,
) -> ApprovedFusedProductionFamily {
  self.residual
}

///|
pub fn ApprovedFusedProductionManifest::target(
  self : ApprovedFusedProductionManifest,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn ApprovedFusedProductionManifest::physical_evidence_digest(
  self : ApprovedFusedProductionManifest,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.physical
}

///|
pub fn ApprovedFusedProductionManifest::physical_artifact_seal_digest(
  self : ApprovedFusedProductionManifest,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.physical_artifact_seal
}

///|
pub fn ApprovedFusedProductionManifest::physical_evidence_seal_digest(
  self : ApprovedFusedProductionManifest,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.physical_evidence_seal
}

///|
pub fn ApprovedFusedProductionManifest::approval_record_digest(
  self : ApprovedFusedProductionManifest,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.approval_record
}

///|
pub fn ApprovedFusedProductionManifest::approved_source_identity(
  self : ApprovedFusedProductionManifest,
) -> @luna_capability_manifest.LunaApprovedImmutableSourceIdentity {
  self.approved_source
}

///|
pub fn ApprovedFusedProductionManifest::approved_key_identity(
  self : ApprovedFusedProductionManifest,
) -> @luna_capability_manifest.LunaSignatureKeyIdentityDigest {
  self.approved_key
}

///|
pub fn ApprovedFusedProductionManifest::detached_signature_digest(
  self : ApprovedFusedProductionManifest,
) -> @luna_capability_manifest.LunaDetachedSignatureDigest {
  self.detached_signature
}

///|
pub fn ApprovedFusedProductionManifest::external_approval_record(
  self : ApprovedFusedProductionManifest,
) -> @luna_capability_manifest.LunaAdmittedExternalRecord {
  self.approval
}

///|
pub fn ApprovedFusedProductionManifest::canonical_bytes(
  self : ApprovedFusedProductionManifest,
) -> Bytes {
  self.canonical
}

///|
pub fn ApprovedFusedProductionManifest::digest(
  self : ApprovedFusedProductionManifest,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.digest
}

///|
pub fn ApprovedFusedProductionManifest::manifest_bindable(
  _self : ApprovedFusedProductionManifest,
) -> Bool {
  true
}

///|
pub fn ApprovedFusedProductionManifest::production_authorized(
  _self : ApprovedFusedProductionManifest,
) -> Bool {
  true
}
