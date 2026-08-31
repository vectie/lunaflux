///|
pub fn AttentionMultiLaunchLimits::maximum_workspace_bytes(
  self : AttentionMultiLaunchLimits,
) -> Int64 {
  self.maximum_workspace_bytes
}

///|
pub fn AttentionDecodeWorkspaceContract::scalar(
  self : AttentionDecodeWorkspaceContract,
) -> AttentionWorkspaceScalar {
  self.scalar
}

///|
pub fn AttentionDecodeWorkspaceContract::split_count(
  self : AttentionDecodeWorkspaceContract,
) -> Int {
  self.split_count
}

///|
pub fn AttentionDecodeWorkspaceContract::partial_count(
  self : AttentionDecodeWorkspaceContract,
) -> Int64 {
  self.partial_count
}

///|
pub fn AttentionDecodeWorkspaceContract::region_offsets_bytes(
  self : AttentionDecodeWorkspaceContract,
) -> (Int64, Int64, Int64) {
  (
    self.maximum_offset_bytes,
    self.denominator_offset_bytes,
    self.numerator_offset_bytes,
  )
}

///|
pub fn AttentionDecodeWorkspaceContract::total_bytes(
  self : AttentionDecodeWorkspaceContract,
) -> Int64 {
  self.total_bytes
}

///|
pub fn AttentionDecodeWorkspaceContract::alignment(
  self : AttentionDecodeWorkspaceContract,
) -> Int64 {
  self.alignment
}

///|
pub fn AttentionOrderedLaunchContract::ordinal(
  self : AttentionOrderedLaunchContract,
) -> Int {
  self.ordinal
}

///|
pub fn AttentionOrderedLaunchContract::stage(
  self : AttentionOrderedLaunchContract,
) -> AttentionDecodeLaunchStage {
  self.stage
}

///|
pub fn AttentionOrderedLaunchContract::entry_point(
  self : AttentionOrderedLaunchContract,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn AttentionOrderedLaunchContract::module_digest(
  self : AttentionOrderedLaunchContract,
) -> @catalog.AotArtifactDigest {
  self.entry_point.family().module_digest()
}

///|
pub fn AttentionOrderedLaunchContract::function_symbol(
  self : AttentionOrderedLaunchContract,
) -> String {
  self.function_symbol
}

///|
pub fn AttentionOrderedLaunchContract::dimensions(
  self : AttentionOrderedLaunchContract,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn AttentionOrderedLaunchContract::operands(
  self : AttentionOrderedLaunchContract,
) -> ReadOnlyArray[@launch_contract.AotOperand] {
  self.operands
}

///|
pub fn AttentionDecodeMultiLaunchContract::operation_id(
  self : AttentionDecodeMultiLaunchContract,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn AttentionDecodeMultiLaunchContract::profile(
  self : AttentionDecodeMultiLaunchContract,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn AttentionDecodeMultiLaunchContract::target(
  self : AttentionDecodeMultiLaunchContract,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn AttentionDecodeMultiLaunchContract::workspace(
  self : AttentionDecodeMultiLaunchContract,
) -> AttentionDecodeWorkspaceContract {
  self.workspace
}

///|
pub fn AttentionDecodeMultiLaunchContract::launches(
  self : AttentionDecodeMultiLaunchContract,
) -> ReadOnlyArray[AttentionOrderedLaunchContract] {
  self.launches
}
