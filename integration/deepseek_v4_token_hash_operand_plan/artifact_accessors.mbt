///|
pub fn DeepSeekV4TokenHashArtifactBinding::status(
  _self : Self,
) -> DeepSeekV4TokenHashArtifactBindingStatus {
  InertUnqualified
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::layer_index(self : Self) -> Int {
  self.layer_index
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::operand_plan_digest(
  self : Self,
) -> DeepSeekV4TokenHashOperandPlanDigest {
  self.operand_plan_digest
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::artifact_admission_digest(
  self : Self,
) -> @advanced_decoder_artifact_admission.AdvancedDecoderArtifactAdmissionDigest {
  self.artifact_admission_digest
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::module_digest(
  self : Self,
) -> @catalog.AotArtifactDigest {
  self.module_digest
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::module_byte_count(
  self : Self,
) -> Int {
  self.module_byte_count
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::entry_point(
  self : Self,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::operand_count(self : Self) -> Int {
  self.operand_count
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::token_hash_operand_ordinal(
  self : Self,
) -> Int {
  self.token_hash_operand_ordinal
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::token_hash_device_offset(
  self : Self,
) -> Int64 {
  self.token_hash_device_offset
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::digest(
  self : Self,
) -> DeepSeekV4TokenHashArtifactBindingDigest {
  self.digest
}

///|
pub fn DeepSeekV4TokenHashArtifactBindingDigest::as_hex(self : Self) -> String {
  let DeepSeekV4TokenHashArtifactBindingDigest(hex) = self
  hex
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::require_compilation_provenance(
  _self : Self,
) -> Unit raise DeepSeekV4TokenHashArtifactBindingError {
  raise CompilationProvenanceUnavailable
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::require_loader_authority(
  _self : Self,
) -> Unit raise DeepSeekV4TokenHashArtifactBindingError {
  raise LoaderAuthorityUnavailable
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::require_launch_authority(
  _self : Self,
) -> Unit raise DeepSeekV4TokenHashArtifactBindingError {
  raise LaunchAuthorityUnavailable
}

///|
pub fn DeepSeekV4TokenHashArtifactBinding::require_physical_qualification(
  _self : Self,
) -> Unit raise DeepSeekV4TokenHashArtifactBindingError {
  raise PhysicalQualificationUnavailable
}
