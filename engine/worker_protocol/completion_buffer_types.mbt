///|
/// Startup-allocated, single-writer worker-to-scheduler completion ingress.
/// Scalar columns avoid constructing heap-backed `CompletionEntry` values on
/// the successful production path. The immutable `CompletionRecord` remains
/// the allocating convenience and fixture API.
pub struct CompletionBuffer {
  priv limits : WorkerProtocolLimits
  priv mut state : CompletionBufferState
  priv mut epoch : UInt64
  priv mut sequence_value : UInt64
  priv mut model_plan_generation_value : UInt64
  priv mut entry_count : Int
  priv mut writer_open : Bool
  priv slot_to_entry : FixedArray[Int]
  priv slots : FixedArray[CompletionSlot]
  priv request_ids : FixedArray[@inference.RequestId]
  priv request_generations : FixedArray[@inference.RequestGeneration]
  priv kinds : FixedArray[CompletionEntryKind]
  priv processed_token_counts : FixedArray[Int]
  priv token_ids : FixedArray[Int]
  priv failures : FixedArray[WorkerFailure]
}

///|
/// Payload-safe scalar outcome category for reusable completion ingress.
pub(all) enum CompletionEntryKind {
  PrefillCompletion
  FinalPrefillCompletion
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

///|
/// Worker-only write lease for one exact completion-buffer and submitted-plan
/// epoch. It can append and submit results, but cannot consume, reset, retire,
/// or otherwise recover either mutable owner.
#valtype
pub struct CompletionWriter(
  CompletionBuffer,
  UInt64,
  SchedulePlanBuffer,
  UInt64
)
