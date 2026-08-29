///|
pub(all) enum LunaTileLimitRule {
  TensorCount
  InstructionCount
  Rank
  LoopRank
  ConstraintCount
  CanonicalBytes
  SharedMemoryBytes
} derive(Debug, Eq)

///|
pub(all) enum LunaTileValidationRule {
  Identifier
  SemanticContract
  TensorOrder
  TensorShape
  TensorStride
  TensorAlignment
  TensorOffset
  TensorExtent
  TensorOverlap
  AffineExpression
  RegionRank
  RegionBounds
  CopyShape
  CopyDType
  CopyAddressSpace
  PipelineStage
  PipelineBarrier
  MmaShape
  MmaDType
  OperationAddressSpace
  Reduction
  Elementwise
  Constraint
  CanonicalEncoding
  AotPlanningPolicy
  CudaLowering
  ParallelPolicy
  ParallelMapping
  ParallelLifetime
  ParallelPipeline
  TensorCoreQualification
} derive(Debug, Eq)

///|
pub(all) suberror LunaTileError {
  InvalidLimit(LunaTileLimitRule)
  LimitExceeded(LunaTileLimitRule)
  InvalidProgram(LunaTileValidationRule)
  InvalidTensor(LunaTileTensorId, LunaTileValidationRule)
  InvalidInstruction(index~ : Int, rule~ : LunaTileValidationRule)
  SizeOverflow(LunaTileValidationRule)
} derive(Debug, Eq)
