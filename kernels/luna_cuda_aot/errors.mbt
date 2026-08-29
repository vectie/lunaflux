///|
pub(all) enum LunaCudaAotRule {
  Limits
  Program
  PlanningInput
  DeviceTarget
  ExecutionShape
  ElementCount
  CompilerPolicy
  Source
  Recipe
  Toolchain
  CompileReceipt
  Determinism
  Artifact
  Evidence
  Specialization
} derive(Debug, Eq)

///|
pub(all) suberror LunaCudaAotError {
  Invalid(LunaCudaAotRule)
  Unsupported(LunaCudaAotRule)
  Mismatch(LunaCudaAotRule)
  LimitExceeded(LunaCudaAotRule)
  SizeOverflow(LunaCudaAotRule)
} derive(Debug, Eq)
