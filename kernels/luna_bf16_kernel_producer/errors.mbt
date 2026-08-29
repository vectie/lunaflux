///|
pub(all) enum LunaBf16KernelProducerRule {
  Limits
  CompleteGraph
  OperationCoverage
  OperationIdentity
  OperationKind
  DeviceTarget
  Profile
  CompilerOutput
  ReusableFamily
  Catalog
  LaunchContracts
  FinalBinding
  Bundle
  Workspace
  CandidateSet
  CandidateSource
  CandidateRecipe
  CandidateInventory
} derive(Debug, Eq)

///|
pub(all) suberror LunaBf16KernelProducerError {
  Invalid(LunaBf16KernelProducerRule)
  Mismatch(LunaBf16KernelProducerRule)
  LimitExceeded(LunaBf16KernelProducerRule, maximum~ : Int)
} derive(Debug, Eq)
