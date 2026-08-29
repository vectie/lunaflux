///|
pub(all) enum LunaFusedCandidateExportStage {
  Qualification
  OutputPath
  Create
  Write
  Publish
  Cleanup
} derive(Debug, Eq)

///|
pub(all) suberror LunaFusedCandidateExportError {
  FusedCandidateExportFailed(LunaFusedCandidateExportStage)
  FusedCandidateExportFailedAndCleanup(
    LunaFusedCandidateExportStage,
    LunaFusedCandidateExportStage
  )
} derive(Debug, Eq)

///|
/// Opaque compiler input for the two candidate-only fused families. It owns no
/// compiled artifact, launch contract, device, runtime, or promotion authority.
pub struct LunaFusedCandidateQualificationSet {
  priv qkv : @luna_cuda_fused_parallel_aot.FusedQkvRopeKvWriteCudaCandidate
  priv residual_norm : @luna_cuda_fused_parallel_aot.FusedResidualRmsNormCudaCandidate
  priv qkv_fallback : @luna_cuda_projection_aot.ProjectionCudaLowering
  priv rope_fallback : @luna_cuda_pointwise_aot.PointwiseCudaLowering
  priv kv_write_fallback : @luna_cuda_pointwise_aot.PointwiseCudaLowering
  priv qkv_key : String
  priv residual_norm_key : String
  priv declaration : Bytes
  priv declaration_sha256 : String
  priv inventory : Bytes
  priv inventory_sha256 : String
}

///|
pub fn LunaFusedCandidateQualificationSet::qkv_fallback(
  self : LunaFusedCandidateQualificationSet,
) -> @luna_cuda_projection_aot.ProjectionCudaLowering {
  self.qkv_fallback
}

///|
pub fn LunaFusedCandidateQualificationSet::rope_fallback(
  self : LunaFusedCandidateQualificationSet,
) -> @luna_cuda_pointwise_aot.PointwiseCudaLowering {
  self.rope_fallback
}

///|
pub fn LunaFusedCandidateQualificationSet::kv_write_fallback(
  self : LunaFusedCandidateQualificationSet,
) -> @luna_cuda_pointwise_aot.PointwiseCudaLowering {
  self.kv_write_fallback
}

///|
pub fn LunaFusedCandidateQualificationSet::qkv_candidate(
  self : LunaFusedCandidateQualificationSet,
) -> @luna_cuda_fused_parallel_aot.FusedQkvRopeKvWriteCudaCandidate {
  self.qkv
}

///|
pub fn LunaFusedCandidateQualificationSet::residual_norm_candidate(
  self : LunaFusedCandidateQualificationSet,
) -> @luna_cuda_fused_parallel_aot.FusedResidualRmsNormCudaCandidate {
  self.residual_norm
}

///|
pub fn LunaFusedCandidateQualificationSet::declaration_bytes(
  self : LunaFusedCandidateQualificationSet,
) -> Bytes {
  self.declaration
}

///|
pub fn LunaFusedCandidateQualificationSet::declaration_sha256(
  self : LunaFusedCandidateQualificationSet,
) -> String {
  self.declaration_sha256
}

///|
pub fn LunaFusedCandidateQualificationSet::inventory_bytes(
  self : LunaFusedCandidateQualificationSet,
) -> Bytes {
  self.inventory
}

///|
pub fn LunaFusedCandidateQualificationSet::inventory_sha256(
  self : LunaFusedCandidateQualificationSet,
) -> String {
  self.inventory_sha256
}

///|
pub fn LunaFusedCandidateQualificationSet::manifest_bindable(
  _self : LunaFusedCandidateQualificationSet,
) -> Bool {
  false
}

///|
/// Root-free receipt for one no-overwrite qualification envelope.
pub struct LunaFusedCandidateExportEvidence {
  priv declaration_sha256 : String
  priv inventory_sha256 : String
  priv export_record_sha256 : String
}

///|
pub fn LunaFusedCandidateExportEvidence::declaration_sha256(
  self : LunaFusedCandidateExportEvidence,
) -> String {
  self.declaration_sha256
}

///|
pub fn LunaFusedCandidateExportEvidence::inventory_sha256(
  self : LunaFusedCandidateExportEvidence,
) -> String {
  self.inventory_sha256
}

///|
pub fn LunaFusedCandidateExportEvidence::export_record_sha256(
  self : LunaFusedCandidateExportEvidence,
) -> String {
  self.export_record_sha256
}
