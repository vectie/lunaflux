///|
pub(all) enum AdvancedDecoderOperandRegionPlanStatus {
  InertUnqualified
} derive(Debug, Eq)

///|
pub struct AdvancedDecoderOperandRegionInput {
  priv operand_ordinal : Int
  priv role : @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperandRole
  priv allocation_identity_sha256 : String
  priv device_offset : Int64
  priv allocation_capacity_bytes : Int64
}

///|
pub struct AdvancedDecoderOperandRegionPlanLimits {
  priv maximum_operands : Int
  priv maximum_allocation_bytes : Int64
  priv maximum_canonical_bytes : Int
}

///|
pub struct AdvancedDecoderOperandRegion {
  priv operand_ordinal : Int
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
  OperandOrdinalMismatch(Int)
  OperandRoleMismatch(Int)
  OperandAlignmentMismatch(Int)
  AllocationCapacityMismatch(Int)
  RegionUndersized(Int)
  RegionOverlap(Int, Int)
  CanonicalEncodingTooLarge
  LaunchAuthorityUnavailable
} derive(Debug, Eq)
