///|
/// Identity-bearing startup input that did not match the model plan.
pub(all) enum IdentitySource {
  WeightContent
  WeightPlan
  WeightNumericSchema
  KernelCatalog
} derive(Debug, Eq)

///|
/// Fixed startup-planning bound exceeded by otherwise validated input.
pub(all) enum PlanBound {
  OperationCount
  InputsPerOperation
  TotalInputCount
} derive(Debug, Eq)

///|
/// Structural rule violated by a resolved device-weight region.
pub(all) enum WeightRegionRule {
  ReferenceMismatch
  NegativeOffset
  NonPositiveLength
  MisalignedOffset
  NonCanonicalOffset
  EndOverflow
  OutsideArena
  OverlappingPrevious
  ByteLengthMismatch
} derive(Debug, Eq)

///|
/// Structural rule violated by the complete immutable weight arena.
pub(all) enum WeightArenaRule {
  NonPositiveSize
  InvalidAlignment
  EmptyRegionSet
  RegionCount
  MaterializedSizeMismatch
  TerminalExtentMismatch
} derive(Debug, Eq)

///|
/// Exact catalog evidence required by one static semantic operation.
pub(all) enum KernelBindingRule {
  SemanticVersion
  NumericDigest
  AotEntryPoint
  LegacyNumericDigest
} derive(Debug, Eq)

///|
/// Structural rule violated by a kernel workspace binding or summary.
pub(all) enum WorkspaceRule {
  InvalidAlignment
  NegativeSize
  MisalignedSize
  InsufficientForOperation
  ExceedsModelBound
  SummarySize
  SummaryAlignment
} derive(Debug, Eq)

///|
/// Bounded, payload-safe startup errors for static device planning.
///
/// Payloads contain only semantic identifiers and fixed enums. They never
/// carry file paths, device pointers, raw model data, or vendor diagnostics.
pub(all) suberror DevicePlanError {
  IdentityMismatch(IdentitySource)
  BoundExceeded(PlanBound)
  InvalidWeightArena(WeightArenaRule)
  MissingLayoutRegion(@plan.TensorRef)
  InvalidLayoutRegion(@plan.TensorRef, WeightRegionRule)
  BindingCountMismatch
  BindingOrderMismatch(@plan.OperationId)
  BindingCapabilityMismatch(@plan.OperationId)
  NumericBindingCountMismatch
  NumericBindingOrderMismatch(@plan.OperationId)
  InvalidKernelBinding(@plan.OperationId, KernelBindingRule)
  MissingWeightRegion(@plan.OperationId, @plan.TensorRef)
  WorkspaceMismatch(@plan.OperationId, WorkspaceRule)
  WorkspaceOverflow(@plan.OperationId)
  WorkspaceSummaryMismatch(WorkspaceRule)
} derive(Debug, Eq)
