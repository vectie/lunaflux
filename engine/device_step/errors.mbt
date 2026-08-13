///|
pub(all) enum DeviceStepLimitRule {
  Rows
  Tokens
  Pages
  SequenceTokens
  TokenId
  ByteCapacity
} derive(Debug, Eq)

///|
pub(all) enum DeviceStepPlanRule {
  ModelGeneration
  PlanSequence
  Counts
  TokenPacking
  PagePacking
  TokenId
  TokenPosition
  SequenceLength
  PageCount
  PageIndex
  PageIdStructure
  CapabilityRecipe
} derive(Debug, Eq)

///|
/// Bounded startup-admission rules for inert paged-KV execution metadata.
pub(all) enum PagedKvBlueprintRule {
  Identity
  Target
  CatalogVersion
  Layout
  Profile
  OperationCount
  OperationOrder
  ContractSemantics
  OperandCount
  OperandRole
  OperandSize
  OperandAlignment
  ActivationRegion
  WorkspaceRegion
  PersistentKvRegion
  ArtifactEntryPoint
  ArtifactModule
} derive(Debug, Eq)

///|
pub(all) enum DeviceStepLifecycle {
  Ready
  Staged
  Poisoned
  ClosingFailed
  Closed
} derive(Debug, Eq)

///|
/// Stable allocation, upload, and cleanup stages.
pub(all) enum DeviceStepStage {
  Counts
  TokenIds
  TokenPositions
  QueryRowOffsets
  SequenceLengths
  PageOffsets
  PhysicalPageIndices
} derive(Debug, Eq)

///|
/// Payload-safe failures for reusable device-step staging.
pub(all) suberror DeviceStepError {
  InvalidLimit(DeviceStepLimitRule)
  InvalidModelPlan
  InvalidLayout
  InvalidBlueprint(PagedKvBlueprintRule)
  InvalidBlueprintOperation(@plan.OperationId, PagedKvBlueprintRule)
  InvalidPlan(DeviceStepPlanRule)
  InvalidLifecycle(
    expected~ : DeviceStepLifecycle,
    actual~ : DeviceStepLifecycle
  )
  StaleStage
  StageOwnerMismatch
  StageEpochOverflow
  AllocationSizeMismatch(DeviceStepStage)
  Protocol(@worker_protocol.WorkerProtocolError)
  Device(DeviceStepStage, @device.DeviceError)
} derive(Debug, Eq)
