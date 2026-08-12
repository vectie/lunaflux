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
  OperandCountExceeded(operation~ : @plan.OperationId, maximum~ : Int)
  DuplicateProfile(KernelProfileId)
  UnknownProfile(KernelProfileId)
  IdentityMismatch
  BindingCountMismatch
  BindingOrderMismatch(@plan.OperationId)
  DuplicateContract(KernelProfileId, @plan.OperationId)
  MissingContract(KernelProfileId, @plan.OperationId)
  ContractForVendorKernel(KernelProfileId, @plan.OperationId)
  KernelFamilyMismatch(KernelProfileId, @plan.OperationId)
} derive(Debug, Eq)
