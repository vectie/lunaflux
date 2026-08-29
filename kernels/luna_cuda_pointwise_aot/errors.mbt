///|
pub(all) enum PointwiseAotRule {
  Limits
  Operation
  Shape
  RuntimeInputs
  PersistentState
  DeviceTarget
  CompilerPolicy
  OperandOrder
  OperandSize
  OperandAlignment
  LaunchDimensions
  Source
  Recipe
  Digest
  KvLayout
} derive(Debug, Eq)

///|
pub(all) suberror PointwiseAotError {
  Invalid(PointwiseAotRule)
  Unsupported(PointwiseAotRule)
  Mismatch(PointwiseAotRule)
  LimitExceeded(PointwiseAotRule)
  SizeOverflow(PointwiseAotRule)
} derive(Debug, Eq)
