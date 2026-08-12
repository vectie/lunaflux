///|
/// Invalid bounded startup input supplied to the activation arena planner.
pub(all) enum DeviceMemoryLimitRule {
  ActivationAlignment
  ArenaBytes
} derive(Debug, Eq)

///|
/// Structural property required from the immutable static device plan.
pub(all) enum DeviceMemoryPlanRule {
  EmptyOperations
  OperationCount
  OperationOrder
  OutputArity
  OutputOrder
  ActivationInput
  ActivationInputCount
  ShapeMismatch
  FinalOutput
  Workspace
  ShapeEnvelope
  TargetDtype
} derive(Debug, Eq)

///|
/// Stage retained if deterministic cleanup itself fails.
pub(all) enum DeviceMemoryFailureStage {
  AllocationValidation
} derive(Debug, Eq)

///|
/// Bounded, payload-safe startup failures for activation/workspace memory.
///
/// Payloads contain only semantic identifiers, fixed rules, and the device
/// facade's bounded error vocabulary. They never contain model data, paths,
/// raw pointers, or vendor diagnostics.
pub(all) suberror DeviceMemoryError {
  InvalidLimit(DeviceMemoryLimitRule)
  InvalidPlan(DeviceMemoryPlanRule)
  InvalidOperation(@plan.OperationId, DeviceMemoryPlanRule)
  InvalidActivation(@plan.ActivationRef, DeviceMemoryPlanRule)
  SizeOverflow(@plan.OperationId)
  ArenaLimitExceeded
  AllocationSizeMismatch
  DeviceFailed(@device.DeviceError)
  CleanupFailed(DeviceMemoryFailureStage, @device.DeviceError)
} derive(Debug, Eq)
