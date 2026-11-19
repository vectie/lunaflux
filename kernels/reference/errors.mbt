///|
/// Caller-supplied allocation bound rejected by the reference executor.
pub(all) enum ReferenceLimitField {
  MaxRows
  MaxColumns
  MaxElements
} derive(Debug, Eq)

///|
/// Semantic shape relationship rejected before numerical work begins.
pub(all) enum ReferenceShapeRule {
  PositiveDimensions
  ElementCount
  EmbeddingTokenCount
  EmbeddingTokenId
  NormalizationWeight
  ProjectionInputWidth
  QkvInputWidth
  QkvOutputWidth
  RotaryLayout
  AttentionLayout
  AttentionHeadRelationship
  ResidualOperands
  MlpGateWeight
  MlpUpWeight
  MlpDownWeight
  Bfloat16ByteCount
} derive(Debug, Eq)

///|
/// Numerical role whose non-finite value was rejected.
pub(all) enum ReferenceValueRole {
  ActivationValue
  TensorValue
  Bfloat16Value
  Epsilon
  RopeTheta
  ComputedValue
} derive(Debug, Eq)

///|
/// Bounded, payload-safe failures from deterministic correctness kernels.
pub(all) suberror ReferenceKernelError {
  InvalidLimit(ReferenceLimitField)
  LimitExceeded(ReferenceLimitField)
  InvalidShape(ReferenceShapeRule)
  InvalidScalar(ReferenceValueRole)
  InvalidIndex
  InvalidPosition
  ArithmeticOverflow
  NonFinite(ReferenceValueRole)
} derive(Debug, Eq)

///|
pub extend ReferenceLimitField with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ReferenceLimitField with Eq::{not_equal, equal}

///|
pub extend ReferenceShapeRule with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ReferenceShapeRule with Eq::{not_equal, equal}

///|
pub extend ReferenceValueRole with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ReferenceValueRole with Eq::{not_equal, equal}

///|
pub extend ReferenceKernelError with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend ReferenceKernelError with Eq::{not_equal, equal}
