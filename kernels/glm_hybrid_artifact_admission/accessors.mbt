///|
pub fn GlmHybridLaunchOperand::role(self : Self) -> GlmHybridLaunchOperandRole {
  self.role
}

///|
pub fn GlmHybridLaunchOperand::byte_count(self : Self) -> Int64 {
  self.byte_count
}

///|
pub fn GlmHybridLaunchOperand::alignment(self : Self) -> Int64 {
  self.alignment
}

///|
pub fn GlmHybridAdmittedArtifact::ordinal(self : Self) -> Int {
  self.ordinal
}

///|
pub fn GlmHybridAdmittedArtifact::operation(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridSemanticOperation {
  self.operation
}

///|
pub fn GlmHybridAdmittedArtifact::capability(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridAotCapability {
  self.capability
}

///|
pub fn GlmHybridAdmittedArtifact::module_digest(
  self : Self,
) -> @catalog.AotArtifactDigest {
  self.module_digest
}

///|
pub fn GlmHybridAdmittedArtifact::module_byte_length(self : Self) -> Int {
  self.module_byte_length
}

///|
pub fn GlmHybridAdmittedArtifact::entry_point(
  self : Self,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn GlmHybridAdmittedArtifact::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn GlmHybridAdmittedArtifact::launch(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.launch
}

///|
pub fn GlmHybridAdmittedArtifact::operands(
  self : Self,
) -> ReadOnlyArray[GlmHybridLaunchOperand] {
  self.operands
}

///|
pub fn GlmHybridAdmittedArtifact::declared_qualification(
  self : Self,
) -> GlmHybridDeclaredQualificationDigest? {
  self.declared_qualification
}

///|
pub fn GlmHybridArtifactAdmissionDigest::as_hex(self : Self) -> String {
  self.hex
}

///|
pub fn GlmHybridArtifactAdmission::source_digest(
  self : Self,
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementsDigest {
  self.source_digest
}

///|
pub fn GlmHybridArtifactAdmission::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn GlmHybridArtifactAdmission::target(self : Self) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn GlmHybridArtifactAdmission::status(
  self : Self,
) -> GlmHybridArtifactAdmissionStatus {
  self.status
}

///|
pub fn GlmHybridArtifactAdmission::artifacts(
  self : Self,
) -> ReadOnlyArray[GlmHybridAdmittedArtifact] {
  self.artifacts
}

///|
pub fn GlmHybridArtifactAdmission::digest(
  self : Self,
) -> GlmHybridArtifactAdmissionDigest {
  self.digest
}

///|
pub fn GlmHybridArtifactAdmission::require_execution_authority(
  self : Self,
) -> Unit raise GlmHybridArtifactAdmissionError {
  ignore(self)
  raise NoExecutionAuthority
}
