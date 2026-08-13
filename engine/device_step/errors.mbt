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
