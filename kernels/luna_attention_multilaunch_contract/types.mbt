///|
pub(all) enum AttentionWorkspaceScalar {
  F32
} derive(Debug, Eq)

///|
pub(all) enum AttentionDecodeLaunchStage {
  Partial
  Merge
} derive(Debug, Eq)

///|
pub struct AttentionMultiLaunchLimits {
  priv maximum_workspace_bytes : Int64
  priv maximum_symbol_bytes : Int
  priv maximum_operands_per_launch : Int
} derive(Debug, Eq)

///|
pub struct AttentionDecodeWorkspaceContract {
  priv scalar : AttentionWorkspaceScalar
  priv split_count : Int
  priv partial_count : Int64
  priv maximum_offset_bytes : Int64
  priv denominator_offset_bytes : Int64
  priv numerator_offset_bytes : Int64
  priv total_bytes : Int64
  priv alignment : Int64
} derive(Debug, Eq)

///|
pub struct AttentionOrderedLaunchInput {
  priv stage : AttentionDecodeLaunchStage
  priv entry_point : @catalog.AotKernelEntryPoint
  priv function_symbol : String
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[@launch_contract.AotOperand]
} derive(Debug, Eq)

///|
pub struct AttentionOrderedLaunchContract {
  priv ordinal : Int
  priv stage : AttentionDecodeLaunchStage
  priv entry_point : @catalog.AotKernelEntryPoint
  priv function_symbol : String
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[@launch_contract.AotOperand]
} derive(Debug, Eq)

///|
pub struct AttentionDecodeMultiLaunchContract {
  priv operation_id : @plan.OperationId
  priv profile : @launch_contract.PagedKernelProfile
  priv target : @catalog.DeviceTarget
  priv workspace : AttentionDecodeWorkspaceContract
  priv launches : ReadOnlyArray[AttentionOrderedLaunchContract]
} derive(Debug, Eq)
