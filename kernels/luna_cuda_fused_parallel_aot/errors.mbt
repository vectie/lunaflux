///|
pub(all) enum FusedParallelAotRule {
  Limits
  Plan
  Operation
  OperationOrder
  Shape
  Profile
  KvLayout
  DeviceTarget
  CompilerPolicy
  ReferenceFallback
  OperandOrder
  OperandSize
  OperandAlignment
  StaticResources
  Source
  Recipe
  Digest
  CompileReceipt
  CompiledArtifact
  Determinism
  KernelAbi
  Evidence
  ObservationOrder
  Correctness
  MicrobenchmarkWin
} derive(Debug, Eq)

///|
pub(all) suberror FusedParallelAotError {
  FusedParallelInvalid(FusedParallelAotRule)
  FusedParallelUnsupported(FusedParallelAotRule)
  FusedParallelMismatch(FusedParallelAotRule)
  FusedParallelLimitExceeded(FusedParallelAotRule)
  FusedParallelOverflow(FusedParallelAotRule)
} derive(Debug, Eq)
