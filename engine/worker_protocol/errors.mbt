///|
/// Public dimensions whose configured bounds are checked at startup.
pub(all) enum ProtocolLimit {
  PrefillRows
  DecodeRows
  PlanRows
  PlanTokens
  PlanPages
  Capabilities
  CompletionSlots
  SequenceTokens
  TokenId
} derive(Debug, Eq)

///|
/// Monotonic identities carried across the scheduler/worker boundary.
pub(all) enum ProtocolIdentity {
  PlanSequenceIdentity
  ModelPlanGenerationIdentity
  PlanBufferEpochIdentity
  PlanRowDraftEpochIdentity
  PlanBuildCheckpointEpochIdentity
  CompletionBufferEpochIdentity
} derive(Debug, Eq)

///|
/// Lifecycle of one reusable schedule-plan buffer. The scheduler owns the
/// mutable buffer; workers receive only authenticated submitted handles.
pub(all) enum PlanBufferState {
  Writable
  Submitted
  Retired
} derive(Debug, Eq)

///|
/// Lifecycle of one reusable worker-to-scheduler completion buffer.
pub(all) enum CompletionBufferState {
  CompletionWritable
  CompletionSubmitted
  Consumed
} derive(Debug, Eq)

///|
/// Immutable flattened tables addressed by row views.
pub(all) enum ProtocolTable {
  Tokens
  Pages
  Capabilities
} derive(Debug, Eq)

///|
/// Worker row categories. The integer in an error is always bounded by the
/// corresponding configured row count.
pub(all) enum WorkerRowKind {
  Prefill
  Decode
} derive(Debug, Eq)

///|
/// Startup recipe validation rules. These categories contain no model or
/// backend data.
pub(all) enum CapabilityRecipeRule {
  EmptyCapabilityRecipe
} derive(Debug, Eq)

///|
/// Exact row validation rules. These names contain no prompt, path, device,
/// model-family, or backend data.
pub(all) enum RowRule {
  EmptyRange
  NegativeRangeStart
  RangeOverflow
  RangeOutsideTable
  NonContiguousRange
  InvalidSequencePosition
  DuplicateRequest
  DuplicateCompletionSlot
  InvalidRowOrder
} derive(Debug, Eq)

///|
/// Payload-safe scheduling-protocol failures.
pub(all) suberror WorkerProtocolError {
  InvalidLimit(ProtocolLimit)
  InvalidIdentity(ProtocolIdentity)
  IdentityOverflow(ProtocolIdentity)
  EmptyPlan
  BoundExceeded(ProtocolLimit)
  TokenBudgetMismatch
  InvalidToken(index~ : Int)
  InvalidCapabilityRecipe(CapabilityRecipeRule)
  InvalidCompletionSlot
  InvalidCompletionToken(slot~ : Int)
  InvalidView(table~ : ProtocolTable, rule~ : RowRule)
  InvalidRowDescriptor(kind~ : WorkerRowKind, rule~ : RowRule)
  InvalidRow(kind~ : WorkerRowKind, index~ : Int, rule~ : RowRule)
  IncompleteTable(ProtocolTable)
  CompletionCountMismatch
  DuplicateCompletionSlot(index~ : Int)
  UnknownCompletionSlot(index~ : Int)
  CompletionIdentityMismatch(index~ : Int)
  CompletionKindMismatch(index~ : Int)
  CompletionTokenCountMismatch(index~ : Int)
  StalePlanSequence
  StaleModelPlanGeneration
  InvalidPlanBufferState(expected~ : PlanBufferState, actual~ : PlanBufferState)
  PlanBufferMismatch
  StalePlanBufferEpoch
  PlanBufferStorageFailure
  PlanRowDraftAlreadyOpen
  PlanRowDraftNotOpen
  PlanRowDraftMismatch
  StalePlanRowDraftEpoch
  PlanBuildCheckpointMismatch
  StalePlanBuildCheckpointEpoch
  InvalidCompletionBufferState(
    expected~ : CompletionBufferState,
    actual~ : CompletionBufferState
  )
  CompletionBufferMismatch
  StaleCompletionBufferEpoch
} derive(Debug, Eq)
