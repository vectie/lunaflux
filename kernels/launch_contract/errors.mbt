///|
pub(all) enum LaunchContractLimit {
  ProfileCount
  ContractCount
  OperandsPerContract
} derive(Debug, Eq)

///|
pub(all) enum ProfileRule {
  NonPositiveId
  NonPositiveDimension
  TokenRowOverflow
  TokenRowMismatch
  OutsideModelEnvelope
} derive(Debug, Eq)

///|
pub(all) enum LaunchDimensionRule {
  InvalidGrid
  InvalidBlock
  TooManyThreads
  InvalidSharedMemory
} derive(Debug, Eq)

///|
pub(all) enum OperandRule {
  InvalidAlignment
  NonPositiveByteCount
  DuplicateRole
  MissingSemanticRole
  UnexpectedSemanticRole
  WorkspaceMismatch
  PositionalRoleMismatch
  RuntimeInputSizeMismatch
  PersistentKvMismatch
  NumericTensorMismatch
  NumericByteCountMismatch
  UnsupportedNumericMetadata
} derive(Debug, Eq)

///|
pub(all) suberror LaunchContractError {
  InvalidLimit(LaunchContractLimit)
  InvalidProfile(ProfileRule)
  InvalidDimensions(LaunchDimensionRule)
  InvalidOperandDefinition(OperandRule)
  InvalidOperand(@plan.OperationId, OperandRule)
  ProfileCountExceeded(maximum~ : Int)
  ContractCountExceeded(maximum~ : Int)
  ContractCountMismatch
  OperandCountExceeded(operation~ : @plan.OperationId, maximum~ : Int)
  DuplicateProfile(KernelProfileId)
  UnknownProfile(KernelProfileId)
  IdentityMismatch
  BindingCountMismatch
  BindingOrderMismatch(@plan.OperationId)
  DuplicateContract(KernelProfileId, @plan.OperationId)
  MissingContract(KernelProfileId, @plan.OperationId)
  ContractForVendorKernel(KernelProfileId, @plan.OperationId)
  ContractForNonPagedKvOperation(KernelProfileId, @plan.OperationId)
  KernelFamilyMismatch(KernelProfileId, @plan.OperationId)
  EntryPointMismatch(KernelProfileId, @plan.OperationId)
  NumericBindingMismatch(@plan.OperationId)
  UnsupportedExecutionContract
  CatalogVersionMismatch
  DeviceKvLayoutMismatch
} derive(Debug, Eq)
