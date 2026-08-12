///|
/// Identifier category rejected while constructing a semantic model plan.
pub(all) enum PlanIdentifierKind {
  Operation
  Tensor
  Activation
  KernelCapability
} derive(Debug, Eq)

///|
/// Shape relationship rejected by model-plan validation.
pub(all) enum PlanShapeRule {
  PositiveDimension
  HiddenHeadRelationship
  AttentionKvHeadRelationship
  DimensionOverflow
  KvCachePositiveDimension
  KvCacheDimensionOverflow
  KvCacheShapeMismatch
  UnsupportedKvCacheDType
  OperationShapeMismatch
} derive(Debug, Eq)

///|
/// Ordered graph invariant rejected by model-plan validation.
pub(all) enum PlanGraphRule {
  EmptyOperationList
  EmptyOperationInputs
  EmptyOperationOutputs
  OperationOrder
  ActivationOutputOrder
  MissingActivationInput
  FinalOutputNotLast
  OperationArity
  OperationInputRoles
  OperationCapabilityMismatch
} derive(Debug, Eq)

///|
/// Bounded, payload-safe failures from semantic model-plan construction.
pub(all) suberror PlanValidationError {
  InvalidIdentifier(PlanIdentifierKind)
  InvalidShape(PlanShapeRule)
  InvalidGraph(PlanGraphRule)
  InvalidWorkspaceBound
  OperationWorkspaceExceedsPlan
} derive(Debug, Eq)
