///|
/// Startup-allocated, single-writer worker-to-scheduler completion ingress.
/// Scalar columns avoid constructing heap-backed `CompletionEntry` values on
/// the successful production path. The immutable `CompletionRecord` remains
/// the allocating convenience and fixture API.
pub struct CompletionBuffer {
  priv limits : WorkerProtocolLimits
  priv mut state : CompletionBufferState
  priv mut epoch : UInt64
  priv mut sequence : PlanSequence
  priv mut model_plan_generation : ModelPlanGeneration
  priv mut entry_count : Int
  priv slots : FixedArray[CompletionSlot]
  priv request_ids : FixedArray[@inference.RequestId]
  priv request_generations : FixedArray[@inference.RequestGeneration]
  priv kinds : FixedArray[CompletionEntryKind]
  priv values : FixedArray[Int]
  priv failures : FixedArray[WorkerFailure]
}

///|
/// Payload-safe scalar outcome category for reusable completion ingress.
pub(all) enum CompletionEntryKind {
  PrefillCompletion
  DecodeCompletion
  FailedCompletion
} derive(Debug, Eq)

///|
/// Immutable scheduler handle for one submitted completion-buffer epoch.
#valtype
pub struct SubmittedCompletion(CompletionBuffer, UInt64)

///|
/// Immutable scalar entry view tied to one submitted completion epoch.
#valtype
pub struct SubmittedCompletionEntry(CompletionBuffer, UInt64, Int)

///|
/// Worker-only proof that the scheduler consumed one submitted epoch.
#valtype
pub struct ConsumedCompletion(CompletionBuffer, UInt64)
