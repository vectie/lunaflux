///|
pub fn JointDiffusionAdmittedModule::digest(
  self : Self,
) -> @catalog.AotArtifactDigest {
  self.digest
}

///|
pub fn JointDiffusionAdmittedModule::module_bytes(self : Self) -> Bytes {
  self.module_bytes
}

///|
pub fn JointDiffusionAdmittedEntryPoint::identity(
  self : Self,
) -> @catalog.AotKernelEntryPoint {
  self.identity
}

///|
pub fn JointDiffusionAdmittedEntryPoint::module_index(self : Self) -> Int {
  self.module_index
}

///|
pub fn JointDiffusionAdmittedEntryPoint::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn JointDiffusionAdmittedLaunch::requirement_ordinal(self : Self) -> Int {
  self.requirement_ordinal
}

///|
pub fn JointDiffusionAdmittedLaunch::operation(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation {
  self.operation
}

///|
pub fn JointDiffusionAdmittedLaunch::capability(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionAotCapability {
  self.capability
}

///|
pub fn JointDiffusionAdmittedLaunch::entry_point(
  self : Self,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn JointDiffusionAdmittedLaunch::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn JointDiffusionAdmittedLaunch::operands(
  self : Self,
) -> ReadOnlyArray[JointDiffusionLaunchOperand] {
  self.operands
}

///|
pub fn JointDiffusionArtifactAdmissionDigest::as_hex(self : Self) -> String {
  self.hex
}

///|
pub fn JointDiffusionArtifactAdmission::schema_version(self : Self) -> Int {
  self.schema_version
}

///|
pub fn JointDiffusionArtifactAdmission::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn JointDiffusionArtifactAdmission::requirements_digest(
  self : Self,
) -> @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn JointDiffusionArtifactAdmission::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn JointDiffusionArtifactAdmission::status(
  self : Self,
) -> JointDiffusionArtifactAdmissionStatus {
  self.status
}

///|
/// Artifact admission never grants module-loading or launch authority.
pub fn JointDiffusionArtifactAdmission::require_execution_authority(
  self : Self,
) -> Unit raise JointDiffusionArtifactAdmissionError {
  ignore(self)
  raise NoExecutionAuthority
}

///|
pub fn JointDiffusionArtifactAdmission::digest(
  self : Self,
) -> JointDiffusionArtifactAdmissionDigest {
  self.digest
}

///|
pub fn JointDiffusionArtifactAdmission::modules(
  self : Self,
) -> ReadOnlyArray[JointDiffusionAdmittedModule] {
  self.modules
}

///|
pub fn JointDiffusionArtifactAdmission::entry_points(
  self : Self,
) -> ReadOnlyArray[JointDiffusionAdmittedEntryPoint] {
  self.entry_points
}

///|
pub fn JointDiffusionArtifactAdmission::launches(
  self : Self,
) -> ReadOnlyArray[JointDiffusionAdmittedLaunch] {
  self.launches
}

///|
pub fn JointDiffusionArtifactAdmission::total_module_bytes(
  self : Self,
) -> Int64 {
  self.total_module_bytes
}
