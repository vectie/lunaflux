///|
fn valid_operand_region_sha256(value : StringView) -> Bool {
  value.length() == 64 &&
  value.all(character => {
    character.is_ascii_digit() || (character >= 'a' && character <= 'f')
  })
}

///|
pub fn AdvancedDecoderOperandRegionInput::new(
  operand_ordinal~ : Int,
  role~ : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole,
  allocation_identity_sha256~ : String,
  device_offset~ : Int64,
  allocation_capacity_bytes~ : Int64,
) -> Self raise AdvancedDecoderOperandRegionPlanError {
  if operand_ordinal < 0 ||
    device_offset < 0L ||
    allocation_capacity_bytes <= 0L {
    raise InvalidRegionInput
  }
  if !valid_operand_region_sha256(allocation_identity_sha256) {
    raise InvalidAllocationIdentity
  }
  {
    operand_ordinal,
    role,
    allocation_identity_sha256,
    device_offset,
    allocation_capacity_bytes,
  }
}

///|
pub fn AdvancedDecoderOperandRegionPlanLimits::new(
  maximum_operands~ : Int,
  maximum_allocation_bytes~ : Int64,
  maximum_canonical_bytes~ : Int,
) -> Self raise AdvancedDecoderOperandRegionPlanError {
  if maximum_operands <= 0 ||
    maximum_operands > 64 ||
    maximum_allocation_bytes <= 0L ||
    maximum_allocation_bytes > 9223372036854775807L ||
    maximum_canonical_bytes <= 0 ||
    maximum_canonical_bytes > 1048576 {
    raise InvalidLimits
  }
  { maximum_operands, maximum_allocation_bytes, maximum_canonical_bytes, }
}
