///|
pub fn AdvancedDecoderAdmittedModule::digest(
  self : Self,
) -> @catalog.AotArtifactDigest {
  self.digest
}

///|
pub fn AdvancedDecoderAdmittedModule::module_bytes(self : Self) -> Bytes {
  self.module_bytes
}

///|
pub fn AdvancedDecoderAdmittedEntryPoint::identity(
  self : Self,
) -> @catalog.AotKernelEntryPoint {
  self.identity
}

///|
pub fn AdvancedDecoderAdmittedEntryPoint::module_index(self : Self) -> Int {
  self.module_index
}

///|
pub fn AdvancedDecoderAdmittedEntryPoint::function_symbol(
  self : Self,
) -> String {
  self.function_symbol
}

///|
pub fn AdvancedDecoderAdmittedLaunch::requirement_ordinal(self : Self) -> Int {
  self.requirement_ordinal
}

///|
pub fn AdvancedDecoderAdmittedLaunch::operation(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation {
  self.operation
}

///|
pub fn AdvancedDecoderAdmittedLaunch::capability(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability {
  self.capability
}

///|
pub fn AdvancedDecoderAdmittedLaunch::entry_point(
  self : Self,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn AdvancedDecoderAdmittedLaunch::module_digest(
  self : Self,
) -> @catalog.AotArtifactDigest {
  self.module_digest
}

///|
pub fn AdvancedDecoderAdmittedLaunch::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn AdvancedDecoderAdmittedLaunch::operands(
  self : Self,
) -> ReadOnlyArray[AdvancedDecoderLaunchOperand] {
  self.operands
}

///|
pub fn AdvancedDecoderAdmittedLaunch::declared_qualification(
  self : Self,
) -> AdvancedDecoderDeclaredQualificationDigest? {
  self.declared_qualification
}

///|
pub fn AdvancedDecoderArtifactAdmissionDigest::as_hex(self : Self) -> String {
  self.hex
}

///|
pub fn AdvancedDecoderArtifactAdmission::schema_version(self : Self) -> Int {
  self.schema_version
}

///|
pub fn AdvancedDecoderArtifactAdmission::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn AdvancedDecoderArtifactAdmission::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn AdvancedDecoderArtifactAdmission::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn AdvancedDecoderArtifactAdmission::status(
  self : Self,
) -> AdvancedDecoderArtifactAdmissionStatus {
  self.status
}

///|
/// Admission never grants module-loading or launch authority.
pub fn AdvancedDecoderArtifactAdmission::require_execution_authority(
  self : Self,
) -> Unit raise AdvancedDecoderArtifactAdmissionError {
  ignore(self)
  raise NoExecutionAuthority
}

///|
pub fn AdvancedDecoderArtifactAdmission::digest(
  self : Self,
) -> AdvancedDecoderArtifactAdmissionDigest {
  self.digest
}

///|
pub fn AdvancedDecoderArtifactAdmission::modules(
  self : Self,
) -> ReadOnlyArray[AdvancedDecoderAdmittedModule] {
  self.modules
}

///|
pub fn AdvancedDecoderArtifactAdmission::entry_points(
  self : Self,
) -> ReadOnlyArray[AdvancedDecoderAdmittedEntryPoint] {
  self.entry_points
}

///|
pub fn AdvancedDecoderArtifactAdmission::launches(
  self : Self,
) -> ReadOnlyArray[AdvancedDecoderAdmittedLaunch] {
  self.launches
}

///|
pub fn AdvancedDecoderArtifactAdmission::total_module_bytes(
  self : Self,
) -> Int64 {
  self.total_module_bytes
}
