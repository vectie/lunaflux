///|
/// Identity-bearing startup input that did not match the model plan.
pub(all) enum IdentitySource {
  WeightLayout
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
  EndOverflow
  OutsideArena
  OverlappingPrevious
} derive(Debug, Eq)

///|
/// Structural rule violated by the complete immutable weight arena.
pub(all) enum WeightArenaRule {
  NonPositiveSize
  InvalidAlignment
  EmptyRegionSet
  RegionCount
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
  MissingWeightRegion(@plan.OperationId, @plan.TensorRef)
  InvalidWeightRegion(@plan.OperationId, @plan.TensorRef, WeightRegionRule)
  WorkspaceMismatch(@plan.OperationId, WorkspaceRule)
  WorkspaceOverflow(@plan.OperationId)
  WorkspaceSummaryMismatch(WorkspaceRule)
} derive(Debug, Eq)
