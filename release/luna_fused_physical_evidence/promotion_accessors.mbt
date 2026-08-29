///|
pub fn FusedManifestApprovalRecord::canonical_bytes(
  self : FusedManifestApprovalRecord,
) -> Bytes {
  self.canonical
}

///|
pub fn FusedManifestApprovalRecord::digest(
  self : FusedManifestApprovalRecord,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.digest
}

///|
pub fn FusedManifestApprovalRecord::manifest_bindable(
  _self : FusedManifestApprovalRecord,
) -> Bool {
  false
}

///|
pub fn FusedManifestApprovalRecord::promotion_authorized(
  _self : FusedManifestApprovalRecord,
) -> Bool {
  false
}

///|
pub fn ApprovedFusedManifestArtifacts::approval_record(
  self : ApprovedFusedManifestArtifacts,
) -> FusedManifestApprovalRecord {
  self.record
}

///|
pub fn ApprovedFusedManifestArtifacts::qkv_compiled_artifact(
  self : ApprovedFusedManifestArtifacts,
) -> @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding {
  self.record.qkv_compiled
}

///|
pub fn ApprovedFusedManifestArtifacts::residual_compiled_artifact(
  self : ApprovedFusedManifestArtifacts,
) -> @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding {
  self.record.residual_compiled
}

///|
pub fn ApprovedFusedManifestArtifacts::approved_source_identity(
  self : ApprovedFusedManifestArtifacts,
) -> @luna_capability_manifest.LunaApprovedImmutableSourceIdentity {
  self.approval.approved_source_identity()
}

///|
pub fn ApprovedFusedManifestArtifacts::approved_key_identity(
  self : ApprovedFusedManifestArtifacts,
) -> @luna_capability_manifest.LunaSignatureKeyIdentityDigest {
  self.approval.approved_key_identity()
}

///|
pub fn ApprovedFusedManifestArtifacts::detached_signature_digest(
  self : ApprovedFusedManifestArtifacts,
) -> @luna_capability_manifest.LunaDetachedSignatureDigest {
  self.approval.detached_signature_digest()
}

///|
pub fn ApprovedFusedManifestArtifacts::external_approval_record(
  self : ApprovedFusedManifestArtifacts,
) -> @luna_capability_manifest.LunaAdmittedExternalRecord {
  self.approval
}

///|
pub fn ApprovedFusedManifestArtifacts::manifest_bindable(
  _self : ApprovedFusedManifestArtifacts,
) -> Bool {
  true
}

///|
pub fn ApprovedFusedManifestArtifacts::promotion_authorized(
  _self : ApprovedFusedManifestArtifacts,
) -> Bool {
  true
}
