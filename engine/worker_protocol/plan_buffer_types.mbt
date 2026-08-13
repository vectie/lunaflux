///|
/// Startup-allocated mutable owner of one reusable scheduler-to-worker plan.
///
/// The scheduler owns this value. A worker receives only a
/// `SubmittedSchedulePlan`, whose every accessor authenticates the buffer
/// epoch and lifecycle before reading. Production schedulers construct exactly
/// two owners, but that overlap policy is deliberately not a protocol global.
/// The owner is single-writer and thread-confined; protocol values are not a
/// synchronization primitive. Append copies each caller `ArrayView` before
/// returning, so callers need only refrain from concurrent source mutation for
/// that call's duration.
///
/// The immutable `SchedulePlan` remains the allocating fixture and convenience
/// API. This buffer is the production no-growth foundation. A reusable
/// completion ingress/ring remains a separate worker-to-scheduler phase gate.
pub struct SchedulePlanBuffer {
  priv limits : WorkerProtocolLimits
  priv mut state : PlanBufferState
  priv mut epoch : UInt64
  priv mut sequence_value : UInt64
  priv mut model_plan_generation_value : UInt64
  priv mut token_budget : Int
  priv mut token_count : Int
  priv mut page_count : Int
  priv mut capability_count : Int
  priv mut prefill_count : Int
  priv mut decode_count : Int
  priv mut draft_open : Bool
  priv mut draft_epoch : UInt64
  priv mut checkpoint_epoch : UInt64
  priv mut checkpoint_token_count : Int
  priv mut checkpoint_page_count : Int
  priv mut checkpoint_capability_count : Int
  priv mut checkpoint_prefill_count : Int
  priv mut checkpoint_decode_count : Int
  priv mut draft_kind : WorkerRowKind
  priv mut draft_request_id : @inference.RequestId
  priv mut draft_request_generation : @inference.RequestGeneration
  priv mut draft_sequence_position : Int
  priv mut draft_produces_token : Bool
  priv mut draft_completion_slot : CompletionSlot
  priv mut draft_sampling : @inference.SamplingParameters
  priv mut draft_sampling_seed_value : UInt64
  priv mut draft_sample_index : Int
  priv mut draft_token_start : Int
  priv mut draft_page_start : Int
  priv mut draft_capability_start : Int
  priv tokens : FixedArray[Int]
  priv pages : @page_allocator.PageIdStorage
  priv capabilities : FixedArray[@plan.KernelCapabilityId]
  priv prefill_request_ids : FixedArray[@inference.RequestId]
  priv prefill_request_generations : FixedArray[@inference.RequestGeneration]
  priv prefill_sequence_starts : FixedArray[Int]
  priv prefill_token_starts : FixedArray[Int]
  priv prefill_token_lengths : FixedArray[Int]
  priv prefill_page_starts : FixedArray[Int]
  priv prefill_page_lengths : FixedArray[Int]
  priv prefill_capability_starts : FixedArray[Int]
  priv prefill_capability_lengths : FixedArray[Int]
  priv prefill_produces_tokens : FixedArray[Bool]
  priv prefill_sampling : FixedArray[@inference.SamplingParameters]
  priv prefill_sampling_seed_values : FixedArray[UInt64]
  priv prefill_sample_indices : FixedArray[Int]
  priv prefill_completion_slots : FixedArray[CompletionSlot]
  priv decode_request_ids : FixedArray[@inference.RequestId]
  priv decode_request_generations : FixedArray[@inference.RequestGeneration]
  priv decode_sequence_positions : FixedArray[Int]
  priv decode_token_starts : FixedArray[Int]
  priv decode_page_starts : FixedArray[Int]
  priv decode_page_lengths : FixedArray[Int]
  priv decode_capability_starts : FixedArray[Int]
  priv decode_capability_lengths : FixedArray[Int]
  priv decode_sampling : FixedArray[@inference.SamplingParameters]
  priv decode_sampling_seed_values : FixedArray[UInt64]
  priv decode_sample_indices : FixedArray[Int]
  priv decode_completion_slots : FixedArray[CompletionSlot]
}

///|
/// Opaque immutable worker handle for one exact submitted buffer epoch.
/// Keeping this handle across retirement or reuse is safe: all later access
/// fails authentication instead of aliasing the new plan.
#valtype
pub struct SubmittedSchedulePlan(SchedulePlanBuffer, UInt64)

///|
/// Scheduler-only proof that one submitted epoch was retired.
#valtype
pub struct RetiredSchedulePlan(SchedulePlanBuffer, UInt64)

///|
/// Opaque immutable prefill-row view tied to its submitted plan epoch.
#valtype
pub struct SubmittedPrefillRow(SchedulePlanBuffer, UInt64, Int)

///|
/// Opaque immutable decode-row view tied to its submitted plan epoch.
#valtype
pub struct SubmittedDecodeRow(SchedulePlanBuffer, UInt64, Int)

///|
/// Scheduler-only authenticated draft for one not-yet-visible row. Scalar
/// pushes mutate reserved table cells; commit publishes the row atomically and
/// rollback clears the reserved logical suffix.
#valtype
pub struct PlanRowDraft(SchedulePlanBuffer, UInt64, UInt64)

///|
/// Authenticated logical snapshot for an in-progress multirow plan build.
/// Only the owning scheduler can restore it while the same plan-buffer epoch
/// remains writable. The handle exposes no payload or model vocabulary.
#valtype
pub struct PlanBuildCheckpoint(SchedulePlanBuffer, UInt64, UInt64)
