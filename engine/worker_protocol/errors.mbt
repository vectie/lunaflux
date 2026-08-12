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
} derive(Debug, Eq)
