///|
pub fn JointDiffusionArtifactAdmissionLimits::new(
  maximum_requirements~ : Int,
  maximum_modules~ : Int,
  maximum_entry_points~ : Int,
  maximum_module_bytes~ : Int64,
  maximum_total_module_bytes~ : Int64,
  maximum_symbol_bytes~ : Int,
  maximum_operands_per_launch~ : Int,
) -> Self raise JointDiffusionArtifactAdmissionError {
  if maximum_requirements <= 0 ||
    maximum_requirements > 64 ||
    maximum_modules <= 0 ||
    maximum_modules > 64 ||
    maximum_entry_points <= 0 ||
    maximum_entry_points > 64 ||
    maximum_module_bytes <= 0L ||
    maximum_module_bytes > 2147483647L ||
    maximum_total_module_bytes <= 0L ||
    maximum_total_module_bytes > 2147483647L ||
    maximum_symbol_bytes <= 0 ||
    maximum_symbol_bytes > 1024 ||
    maximum_operands_per_launch <= 0 ||
    maximum_operands_per_launch > 32 {
    raise InvalidLimit
  }
  {
    maximum_requirements,
    maximum_modules,
    maximum_entry_points,
    maximum_module_bytes,
    maximum_total_module_bytes,
    maximum_symbol_bytes,
    maximum_operands_per_launch,
  }
}

///|
pub fn JointDiffusionModuleInput::new(
  digest~ : @catalog.AotArtifactDigest,
  module_bytes~ : Bytes,
) -> Self {
  { digest, module_bytes, }
}

///|
pub fn JointDiffusionLaunchOperand::new(
  role~ : JointDiffusionLaunchOperandRole,
  byte_count~ : Int64,
  alignment~ : Int64,
) -> Self raise JointDiffusionArtifactAdmissionError {
  if byte_count <= 0L ||
    byte_count > 2147483647L ||
    alignment <= 0L ||
    alignment > 4096L ||
    (alignment & (alignment - 1L)) != 0L {
    raise InvalidLaunchOperands(-1)
  }
  { role, byte_count, alignment, }
}

///|
pub fn JointDiffusionLaunchOperand::role(
  self : Self,
) -> JointDiffusionLaunchOperandRole {
  self.role
}

///|
pub fn JointDiffusionLaunchOperand::byte_count(self : Self) -> Int64 {
  self.byte_count
}

///|
pub fn JointDiffusionLaunchOperand::alignment(self : Self) -> Int64 {
  self.alignment
}

///|
pub fn JointDiffusionLaunchEvidenceInput::new(
  requirements_digest~ : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirementsDigest,
  requirement_ordinal~ : Int,
  operation~ : @joint_diffusion_kernel_requirements.JointDiffusionSemanticOperation,
  capability~ : @joint_diffusion_kernel_requirements.JointDiffusionAotCapability,
  target~ : @catalog.DeviceTarget,
  entry_point~ : @catalog.AotKernelEntryPoint,
  dimensions~ : @launch_contract.AotLaunchDimensions,
  operands~ : ArrayView[JointDiffusionLaunchOperand],
) -> Self raise JointDiffusionArtifactAdmissionError {
  if operands.is_empty() || operands.length() > 32 {
    raise InvalidLaunchOperands(requirement_ordinal)
  }
  {
    requirements_digest,
    requirement_ordinal,
    operation,
    capability,
    target,
    entry_point,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
  }
}
