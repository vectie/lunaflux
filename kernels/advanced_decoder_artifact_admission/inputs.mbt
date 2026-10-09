///|
pub fn AdvancedDecoderArtifactAdmissionLimits::new(
  maximum_requirements~ : Int,
  maximum_modules~ : Int,
  maximum_entry_points~ : Int,
  maximum_module_bytes~ : Int64,
  maximum_total_module_bytes~ : Int64,
  maximum_symbol_bytes~ : Int,
  maximum_operands_per_launch~ : Int,
  maximum_canonical_bytes~ : Int,
) -> Self raise AdvancedDecoderArtifactAdmissionError {
  if maximum_requirements <= 0 ||
    maximum_requirements > 4096 ||
    maximum_modules <= 0 ||
    maximum_modules > 4096 ||
    maximum_entry_points <= 0 ||
    maximum_entry_points > 4096 ||
    maximum_module_bytes <= 0L ||
    maximum_module_bytes > 2147483647L ||
    maximum_total_module_bytes <= 0L ||
    maximum_total_module_bytes > 2147483647L ||
    maximum_symbol_bytes <= 0 ||
    maximum_symbol_bytes > 1024 ||
    maximum_operands_per_launch <= 0 ||
    maximum_operands_per_launch > 32 ||
    maximum_canonical_bytes <= 0 ||
    maximum_canonical_bytes > 16777216 {
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
    maximum_canonical_bytes,
  }
}

///|
pub fn AdvancedDecoderDeclaredQualificationDigest::from_sha256(
  hex : String,
) -> Self raise AdvancedDecoderArtifactAdmissionError {
  if hex.length() != 64 {
    raise InvalidDeclaredQualificationDigest
  }
  for character in hex {
    if !(character >= '0' && character <= '9') &&
      !(character >= 'a' && character <= 'f') {
      raise InvalidDeclaredQualificationDigest
    }
  }
  { hex, }
}

///|
pub fn AdvancedDecoderDeclaredQualificationDigest::as_hex(
  self : Self,
) -> String {
  self.hex
}

///|
pub fn AdvancedDecoderModuleInput::new(
  digest~ : @catalog.AotArtifactDigest,
  module_bytes~ : Bytes,
) -> Self {
  { digest, module_bytes, }
}

///|
pub fn AdvancedDecoderLaunchOperand::new(
  role~ : AdvancedDecoderLaunchOperandRole,
  byte_count~ : Int64,
  alignment~ : Int64,
) -> Self raise AdvancedDecoderArtifactAdmissionError {
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
pub fn AdvancedDecoderLaunchOperand::role(
  self : Self,
) -> AdvancedDecoderLaunchOperandRole {
  self.role
}

///|
pub fn AdvancedDecoderLaunchOperand::byte_count(self : Self) -> Int64 {
  self.byte_count
}

///|
pub fn AdvancedDecoderLaunchOperand::alignment(self : Self) -> Int64 {
  self.alignment
}

///|
pub fn AdvancedDecoderLaunchEvidenceInput::new(
  requirements_digest~ : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest,
  source_identity~ : @spec.ModelIdentity,
  requirement_ordinal~ : Int,
  operation~ : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation,
  capability~ : @advanced_decoder_kernel_requirements.AdvancedDecoderAotCapability,
  target~ : @catalog.DeviceTarget,
  entry_point~ : @catalog.AotKernelEntryPoint,
  module_digest~ : @catalog.AotArtifactDigest,
  dimensions~ : @launch_contract.AotLaunchDimensions,
  operands~ : ArrayView[AdvancedDecoderLaunchOperand],
  declared_qualification? : AdvancedDecoderDeclaredQualificationDigest,
) -> Self raise AdvancedDecoderArtifactAdmissionError {
  if operands.is_empty() || operands.length() > 32 {
    raise InvalidLaunchOperands(requirement_ordinal)
  }
  {
    requirements_digest,
    source_identity,
    requirement_ordinal,
    operation,
    capability,
    target,
    entry_point,
    module_digest,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    declared_qualification,
  }
}
