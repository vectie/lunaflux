///|
/// Identifier category rejected while constructing a semantic model plan.
pub(all) enum PlanIdentifierKind {
  Operation
  Tensor
  Activation
  KernelCapability
  KvLayer
} derive(Debug, Eq)

///|
/// Startup construction envelope rejected before any plan-owned proportional
/// storage is allocated.
pub(all) enum ModelPlanLimitKind {
  Operations
  NumericTensors
  TotalValueInputs
  ValueInputsPerOperation
  RuntimeInputsPerOperation
  OutputsPerOperation
  CanonicalBytes
  IncoherentLimits
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
  KvLayerCoverage
  KvLayerOrder
  KvExecutionContract
  KvExecutionModeMismatch
  MissingTensor
  TensorRoleMismatch
  NumericTensorType
  NumericTensorShape
  UnusedParameterTensor
  UnreferencedNumericMetadata
  NumericOperationBinding
  NumericActivationType
} derive(Debug, Eq)

///|
/// Bounded, payload-safe failures from semantic model-plan construction.
pub(all) suberror PlanValidationError {
  InvalidIdentifier(PlanIdentifierKind)
  InvalidShape(PlanShapeRule)
  InvalidGraph(PlanGraphRule)
  InvalidWorkspaceBound
  OperationWorkspaceExceedsPlan
  InvalidLimit(ModelPlanLimitKind)
  LimitExceeded(ModelPlanLimitKind)
  CanonicalEncodingMismatch
} derive(Debug, Eq)

///|
pub extend PlanIdentifierKind with @debug.Debug::{to_repr}

///|
pub extend PlanIdentifierKind with Eq::{not_equal, equal}

///|
pub extend ModelPlanLimitKind with @debug.Debug::{to_repr}

///|
pub extend ModelPlanLimitKind with Eq::{not_equal, equal}

///|
pub extend PlanShapeRule with @debug.Debug::{to_repr}

///|
pub extend PlanShapeRule with Eq::{not_equal, equal}

///|
pub extend PlanGraphRule with @debug.Debug::{to_repr}

///|
pub extend PlanGraphRule with Eq::{not_equal, equal}

///|
pub extend PlanValidationError with @debug.Debug::{to_repr}

///|
pub extend PlanValidationError with Eq::{not_equal, equal}
