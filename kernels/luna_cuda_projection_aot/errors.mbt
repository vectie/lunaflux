///|
pub(all) enum ProjectionAotRule {
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
} derive(Debug, Eq)

///|
pub(all) suberror ProjectionAotError {
  Invalid(ProjectionAotRule)
  Unsupported(ProjectionAotRule)
  Mismatch(ProjectionAotRule)
  LimitExceeded(ProjectionAotRule)
  SizeOverflow(ProjectionAotRule)
} derive(Debug, Eq)
