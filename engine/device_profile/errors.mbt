///|
/// Which immutable request-shape invariant failed.
pub(all) enum ProfileShapeRule {
  NonPositiveBatchRows
  NonPositiveQueryTokens
  BatchExceedsEnvelope
  QueryExceedsEnvelope
  TokenRowOverflow
} derive(Debug, Eq)

///|
/// Which fact differs between the static execution and memory plans.
pub(all) enum ProfilePlanRule {
  Identity
  Target
  EmptyOperations
  OperationCount
  OperationOrder
  ActivationCount
  MaximumTokenRows
  FinalOutput
  Workspace
} derive(Debug, Eq)

///|
/// Which operation or graph invariant cannot be represented by this profile.
pub(all) enum ProfileOperationRule {
  OutputArity
  OutputOrder
  InputRoles
  ActivationInput
  ActivationInputOrder
  ActivationWidth
  Shape
  MemoryBinding
  Workspace
} derive(Debug, Eq)

///|
/// Which arena-range invariant failed while deriving an exact view.
pub(all) enum ProfileRegionRule {
  Missing
  NegativeOffset
  NegativeLength
  EmptyRequiredRegion
  InvalidAlignment
  MisalignedOffset
  LogicalExceedsReserved
  OutsideArena
  IdentifierOverflow
} derive(Debug, Eq)

///|
/// Payload-safe startup failures for exact device execution profiles.
pub(all) suberror DeviceProfileError {
  InvalidShape(ProfileShapeRule)
  InvalidPlan(ProfilePlanRule)
  InvalidOperation(@plan.OperationId, ProfileOperationRule)
  InvalidRegion(ProfileRegionId, ProfileRegionRule)
  SizeOverflow(@plan.OperationId)
  TokenSizeOverflow
} derive(Debug, Eq)
