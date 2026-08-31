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
  NumericSchema
  NumericPolicy
  DeviceNumericCapability
  OperationNumericDigest
  ScaleRegion
  MixedFp8Execution
  GraphMemory
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
/// Stable native-resource stages for the full paged graph executor.
pub(all) enum PagedGraphResource {
  WeightLease
  Stream
  ActivationWorkspaceArena
  PersistentKvArena
  ActivationReadback
  Module
  Function
  KernelArguments
  OrderedExecutor
  FusedDispatchCanary
  SamplingResult
} derive(Debug, Eq)

///|
pub(all) enum FusedGroupedOperationRule {
  Catalog
  Target
  Family
  Generation
  OperationOrder
  StandaloneFallback
  KernelAbi
  Launch
  GraphCapture
  Resource
  ReadOnlyCatalog
  PagedKvWriteQualification
  ModelIdentity
  Profile
  KvLayout
  Activation
  Toolchain
  Driver
  RuntimeDevice
  StreamOrder
  DispatchAuthority
} derive(Debug, Eq)

///|
pub(all) suberror FusedGroupedOperationError {
  InvalidFusedGroupedOperation(FusedGroupedOperationRule)
  MismatchedFusedGroupedOperation(FusedGroupedOperationRule)
  QkvRequiresReadOnlyAttentionSplit
} derive(Debug, Eq)

///|
/// Bounded executor evidence rejected before dispatch.
pub(all) enum PagedGraphExecutorRule {
  BlueprintScope
  ContextTarget
  WeightIdentity
  WeightLayout
  WeightRegion
  AllocationSize
  ArtifactModule
  LaunchDimensions
  SamplingOutput
  SamplingMode
  SamplingArtifact
  Fp8FrameEnvelope
  Fp8FrameAdmission
  Fp8ScaleEvidence
  GraphBucket
  ResourceState
} derive(Debug, Eq)

///|
/// Single-owner synchronous paged execution lifecycle.
pub(all) enum PagedGraphLifecycle {
  GraphReady
  GraphStaged
  GraphExecuting
  GraphExecuted
  GraphPoisoned
  GraphClosingFailed
  GraphClosed
} derive(Debug, Eq)

///|
/// Startup-only selection for an already admitted immutable launch sequence.
/// The fallback variant is the sole policy that permits capture unavailability
/// to select the ordered eager base.
pub(all) enum PagedGraphExecutionPolicy {
  PagedOrderedEagerOnly
  PagedCapturedRequired
  PagedCapturedWithEagerFallback
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
  DescriptorArgumentMismatch(DeviceStepStage)
  AllocationSizeMismatch(DeviceStepStage)
  InvalidExecutor(PagedGraphExecutorRule)
  InvalidExecutorOperation(@plan.OperationId, PagedGraphExecutorRule)
  InvalidExecutorLifecycle(
    expected~ : PagedGraphLifecycle,
    actual~ : PagedGraphLifecycle
  )
  ExecutorDevice(PagedGraphResource, @device.DeviceError)
  ExecutorLaunchFailed(@plan.OperationId, @device.DeviceError)
  ExecutorSampling(@sampling.SamplingError)
  ExecutorOwnerMismatch
  ExecutorEpochMismatch
  NumericWeightClose
  Protocol(@worker_protocol.WorkerProtocolError)
  Wire(@worker_wire.WorkerWireError)
  Device(DeviceStepStage, @device.DeviceError)
} derive(Debug, Eq)
