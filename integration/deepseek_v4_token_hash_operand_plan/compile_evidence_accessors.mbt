///|
pub fn DeepSeekV4TokenHashCompileEvidenceBinding::status(
  _self : Self,
) -> DeepSeekV4TokenHashCompileEvidenceBindingStatus {
  InertSelfConsistentUnqualified
}

///|
pub fn DeepSeekV4TokenHashCompileEvidenceBinding::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn DeepSeekV4TokenHashCompileEvidenceBinding::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn DeepSeekV4TokenHashCompileEvidenceBinding::layer_index(
  self : Self,
) -> Int {
  self.layer_index
}

///|
pub fn DeepSeekV4TokenHashCompileEvidenceBinding::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn DeepSeekV4TokenHashCompileEvidenceBinding::artifact_binding_digest(
  self : Self,
) -> DeepSeekV4TokenHashArtifactBindingDigest {
  self.artifact_binding_digest
}

///|
pub fn DeepSeekV4TokenHashCompileEvidenceBinding::compile_evidence_digest(
  self : Self,
) -> @cuda_offline_compile_evidence.CudaOfflineCompileEvidenceDigest {
  self.compile_evidence_digest
}

///|
pub fn DeepSeekV4TokenHashCompileEvidenceBinding::source_digest_sha256(
  self : Self,
) -> String {
  self.source_digest_sha256
}

///|
pub fn DeepSeekV4TokenHashCompileEvidenceBinding::recipe_digest_sha256(
  self : Self,
) -> String {
  self.recipe_digest_sha256
}

///|
pub fn DeepSeekV4TokenHashCompileEvidenceBinding::driver_identity_sha256(
  self : Self,
) -> String {
  self.driver_identity_sha256
}

///|
pub fn DeepSeekV4TokenHashCompileEvidenceBinding::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn DeepSeekV4TokenHashCompileEvidenceBinding::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn DeepSeekV4TokenHashCompileEvidenceBinding::module_digest(
  self : Self,
) -> @catalog.AotArtifactDigest {
  self.module_digest
}

///|
pub fn DeepSeekV4TokenHashCompileEvidenceBinding::module_byte_count(
  self : Self,
) -> Int {
  self.module_byte_count
}

///|
pub fn DeepSeekV4TokenHashCompileEvidenceBinding::digest(
  self : Self,
) -> DeepSeekV4TokenHashCompileEvidenceBindingDigest {
  self.digest
}

///|
pub fn DeepSeekV4TokenHashCompileEvidenceBindingDigest::as_hex(
  self : Self,
) -> String {
  let DeepSeekV4TokenHashCompileEvidenceBindingDigest(hex) = self
  hex
}

///|
pub fn DeepSeekV4TokenHashCompileEvidenceBinding::require_compilation_provenance(
  _self : Self,
) -> Unit raise DeepSeekV4TokenHashCompileEvidenceBindingError {
  raise CompilationProvenanceUnavailable
}

///|
pub fn DeepSeekV4TokenHashCompileEvidenceBinding::require_loader_authority(
  _self : Self,
) -> Unit raise DeepSeekV4TokenHashCompileEvidenceBindingError {
  raise LoaderAuthorityUnavailable
}

///|
pub fn DeepSeekV4TokenHashCompileEvidenceBinding::require_launch_authority(
  _self : Self,
) -> Unit raise DeepSeekV4TokenHashCompileEvidenceBindingError {
  raise LaunchAuthorityUnavailable
}

///|
pub fn DeepSeekV4TokenHashCompileEvidenceBinding::require_physical_qualification(
  _self : Self,
) -> Unit raise DeepSeekV4TokenHashCompileEvidenceBindingError {
  raise PhysicalQualificationUnavailable
}
