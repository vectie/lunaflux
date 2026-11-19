///|
pub(all) enum AdvancedDecoderOperandRegionPlanStatus {
  InertUnqualified
} derive(Debug, Eq)

///|
pub struct AdvancedDecoderOperandRegionInput {
  priv operand_ordinal : Int32
  priv role : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole
  priv allocation_identity_sha256 : String
  priv device_offset : Int64
  priv allocation_capacity_bytes : Int64
}

///|
pub struct AdvancedDecoderOperandRegionPlanLimits {
  priv maximum_operands : Int32
  priv maximum_allocation_bytes : Int64
  priv maximum_canonical_bytes : Int32
}

///|
pub struct AdvancedDecoderOperandRegion {
  priv operand_ordinal : Int32
  priv role : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole
  priv byte_count : Int64
  priv alignment : Int64
  priv allocation_identity_sha256 : String
  priv device_offset : Int64
  priv end_offset : Int64
  priv allocation_capacity_bytes : Int64
}

///|
pub struct AdvancedDecoderOperandRegionPlanDigest(String) derive(Debug, Eq, Hash)

///|
pub struct AdvancedDecoderOperandRegionPlan {
  priv regions : ReadOnlyArray[AdvancedDecoderOperandRegion]
  priv digest : AdvancedDecoderOperandRegionPlanDigest
}

///|
pub(all) suberror AdvancedDecoderOperandRegionPlanError {
  InvalidLimits
  InvalidAllocationIdentity
  InvalidRegionInput
  OperandCountMismatch
  OperandOrdinalMismatch(Int32)
  OperandRoleMismatch(Int32)
  OperandAlignmentMismatch(Int32)
  AllocationCapacityMismatch(Int32)
  RegionUndersized(Int32)
  RegionOverlap(Int32, Int32)
  CanonicalEncodingTooLarge
  LaunchAuthorityUnavailable
} derive(Debug, Eq)
