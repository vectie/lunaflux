///|
pub fn CudaOfflineCompileEvidence::status(
  _self : Self,
) -> CudaOfflineCompileEvidenceStatus {
  InertSelfConsistent
}

///|
pub fn CudaOfflineCompileEvidence::source_digest_sha256(self : Self) -> String {
  self.source_digest_sha256
}

///|
pub fn CudaOfflineCompileEvidence::recipe_digest_sha256(self : Self) -> String {
  self.recipe_digest_sha256
}

///|
pub fn CudaOfflineCompileEvidence::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn CudaOfflineCompileEvidence::driver_identity_sha256(
  self : Self,
) -> String {
  self.driver_identity_sha256
}

///|
pub fn CudaOfflineCompileEvidence::target(self : Self) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn CudaOfflineCompileEvidence::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn CudaOfflineCompileEvidence::module_digest(
  self : Self,
) -> @catalog.AotArtifactDigest {
  self.module_digest
}

///|
pub fn CudaOfflineCompileEvidence::module_byte_count(self : Self) -> Int {
  self.module_byte_count
}

///|
pub fn CudaOfflineCompileEvidence::digest(
  self : Self,
) -> CudaOfflineCompileEvidenceDigest {
  self.receipt_digest
}

///|
pub fn CudaOfflineCompileEvidenceDigest::as_hex(self : Self) -> String {
  let CudaOfflineCompileEvidenceDigest(hex) = self
  hex
}

///|
pub fn CudaOfflineCompileEvidence::require_producer_authority(
  _self : Self,
) -> Unit raise CudaOfflineCompileEvidenceError {
  raise ProducerAuthorityUnavailable
}

///|
pub fn CudaOfflineCompileEvidence::require_compiler_execution_proof(
  _self : Self,
) -> Unit raise CudaOfflineCompileEvidenceError {
  raise CompilerExecutionProofUnavailable
}

///|
pub fn CudaOfflineCompileEvidence::require_execution_authority(
  _self : Self,
) -> Unit raise CudaOfflineCompileEvidenceError {
  raise ExecutionAuthorityUnavailable
}
