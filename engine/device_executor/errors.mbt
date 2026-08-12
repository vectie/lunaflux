///|
/// Stable preparation and lifecycle stages. No native pointer, artifact
/// symbol, filesystem path, or vendor diagnostic enters this vocabulary.
pub(all) enum DeviceExecutorStage {
  Stream
  ActivationWorkspaceArena
  TokenStaging
  Module
  Function
  VendorGemmHandle
  VendorGemmPlan
  KernelArguments
  TokenTransfer
  OutputTransfer
  AotLaunch
  VendorGemmRun
} derive(Debug, Eq)

///|
pub(all) enum DeviceExecutorValidationRule {
  Identity
  Target
  CatalogVersion
  ContextCapability
  WeightLayout
  MemoryPlan
  ExecutionProfile
  KernelProfile
  OperationCount
  OperationOrder
  ArtifactModule
  ArtifactEntryPoint
  UnsupportedVendorOperation
  LaunchDimensions
} derive(Debug, Eq)

///|
pub(all) enum DeviceExecutorOperationRule {
  ProfileMissing
  ProfileOrder
  BindingMismatch
  ContractMissing
  ContractMismatch
  OperandMissing
  OperandSize
  OperandAlignment
  OperandSpace
  Workspace
  VendorShape
  VendorInputs
  SizeOverflow
} derive(Debug, Eq)

///|
pub(all) enum DeviceExecutorCleanupCause {
  Device(@device.DeviceError)
  Memory(@device_memory.DeviceMemoryError)
} derive(Debug, Eq)

///|
pub(all) suberror DeviceExecutorError {
  InvalidPlan(DeviceExecutorValidationRule)
  InvalidOperation(@plan.OperationId, DeviceExecutorOperationRule)
  DeviceFailed(DeviceExecutorStage, @device.DeviceError)
  MemoryFailed(DeviceExecutorStage, @device_memory.DeviceMemoryError)
  CleanupFailed(DeviceExecutorStage, DeviceExecutorCleanupCause)
  InvalidTokenBytes(expected~ : Int64, actual~ : Int64)
  InvalidTokenId(index~ : Int)
  InvalidOutputRow(index~ : Int)
  OutputUnavailable
  Closed
} derive(Debug, Eq)
