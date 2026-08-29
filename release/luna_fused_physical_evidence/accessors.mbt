///|
pub fn AdmittedFusedPhysicalEvidence::canonical_bytes(
  self : AdmittedFusedPhysicalEvidence,
) -> Bytes {
  self.canonical
}

///|
pub fn AdmittedFusedPhysicalEvidence::digest(
  self : AdmittedFusedPhysicalEvidence,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.digest
}

///|
pub fn AdmittedFusedPhysicalEvidence::referee_shape_set_digest(
  self : AdmittedFusedPhysicalEvidence,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.referee_shape_set_digest
}

///|
pub fn AdmittedFusedPhysicalEvidence::artifact_directory_seal_digest(
  self : AdmittedFusedPhysicalEvidence,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.artifact_directory_seal_digest
}

///|
pub fn AdmittedFusedPhysicalEvidence::evidence_directory_seal_digest(
  self : AdmittedFusedPhysicalEvidence,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.evidence_directory_seal_digest
}

///|
pub fn AdmittedFusedPhysicalEvidence::approved_policy_digest(
  self : AdmittedFusedPhysicalEvidence,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.approved_policy_digest
}

///|
pub fn AdmittedFusedPhysicalEvidence::qkv_candidate_digest(
  self : AdmittedFusedPhysicalEvidence,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.qkv.candidate_digest
}

///|
pub fn AdmittedFusedPhysicalEvidence::qkv_source_digest(
  self : AdmittedFusedPhysicalEvidence,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.qkv.source_digest
}

///|
pub fn AdmittedFusedPhysicalEvidence::qkv_recipe_digest(
  self : AdmittedFusedPhysicalEvidence,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.qkv.recipe_digest
}

///|
pub fn AdmittedFusedPhysicalEvidence::qkv_cubin_digest(
  self : AdmittedFusedPhysicalEvidence,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.qkv.module_digest
}

///|
pub fn AdmittedFusedPhysicalEvidence::qkv_compile_receipt_digest(
  self : AdmittedFusedPhysicalEvidence,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.qkv.receipt_digest
}

///|
pub fn AdmittedFusedPhysicalEvidence::qkv_observed_absolute_error_ppb(
  self : AdmittedFusedPhysicalEvidence,
) -> Int64 {
  self.qkv.observed_absolute_error_ppb
}

///|
pub fn AdmittedFusedPhysicalEvidence::qkv_observed_relative_error_ppb(
  self : AdmittedFusedPhysicalEvidence,
) -> Int64 {
  self.qkv.observed_relative_error_ppb
}

///|
pub fn AdmittedFusedPhysicalEvidence::qkv_dispatch_canary(
  self : AdmittedFusedPhysicalEvidence,
) -> UInt {
  self.qkv.dispatch_canary
}

///|
pub fn AdmittedFusedPhysicalEvidence::qkv_dispatch_canary_per_token(
  self : AdmittedFusedPhysicalEvidence,
) -> UInt {
  self.qkv.dispatch_canary_per_token
}

///|
pub fn AdmittedFusedPhysicalEvidence::residual_candidate_digest(
  self : AdmittedFusedPhysicalEvidence,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.residual_norm.candidate_digest
}

///|
pub fn AdmittedFusedPhysicalEvidence::residual_source_digest(
  self : AdmittedFusedPhysicalEvidence,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.residual_norm.source_digest
}

///|
pub fn AdmittedFusedPhysicalEvidence::residual_recipe_digest(
  self : AdmittedFusedPhysicalEvidence,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.residual_norm.recipe_digest
}

///|
pub fn AdmittedFusedPhysicalEvidence::residual_cubin_digest(
  self : AdmittedFusedPhysicalEvidence,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.residual_norm.module_digest
}

///|
pub fn AdmittedFusedPhysicalEvidence::residual_compile_receipt_digest(
  self : AdmittedFusedPhysicalEvidence,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.residual_norm.receipt_digest
}

///|
pub fn AdmittedFusedPhysicalEvidence::residual_observed_absolute_error_ppb(
  self : AdmittedFusedPhysicalEvidence,
) -> Int64 {
  self.residual_norm.observed_absolute_error_ppb
}

///|
pub fn AdmittedFusedPhysicalEvidence::residual_observed_relative_error_ppb(
  self : AdmittedFusedPhysicalEvidence,
) -> Int64 {
  self.residual_norm.observed_relative_error_ppb
}

///|
pub fn AdmittedFusedPhysicalEvidence::residual_dispatch_canary(
  self : AdmittedFusedPhysicalEvidence,
) -> UInt {
  self.residual_norm.dispatch_canary
}

///|
pub fn AdmittedFusedPhysicalEvidence::residual_dispatch_canary_per_token(
  self : AdmittedFusedPhysicalEvidence,
) -> UInt {
  self.residual_norm.dispatch_canary_per_token
}

///|
pub fn AdmittedFusedPhysicalEvidence::manifest_bindable(
  _self : AdmittedFusedPhysicalEvidence,
) -> Bool {
  false
}

///|
pub fn AdmittedFusedPhysicalEvidence::promotion_authorized(
  _self : AdmittedFusedPhysicalEvidence,
) -> Bool {
  false
}
