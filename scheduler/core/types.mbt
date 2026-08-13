///|
const SLOT_FREE : Int = 0

///|
const SLOT_WAITING : Int = 1

///|
const SLOT_PREFILL : Int = 2

///|
const SLOT_DECODE : Int = 3

///|
const SLOT_FINISHING : Int = 4

///|
const SLOT_CANCELLED : Int = 5

///|
const SLOT_DEADLINE : Int = 6

///|
const SLOT_FAILED : Int = 7

///|
const SLOT_RETIRED : Int = 8

///|
const MAX_SLOT_GENERATION : Int = 2_147_483_647

///|
const MAX_GENERATION : UInt64 = 0xffffffffffffffffUL

///|
const NO_INDEX : Int = -1

///|
/// Allocation-free result of one scheduling attempt. Submitted identities are
/// opaque and must be resolved through the scheduler that produced them.
#valtype
pub enum BuildNextOutcome {
  BuildIdle
  BuildBackpressured
  BuildSubmitted(Int, UInt64)
}

///|
pub fn BuildNextOutcome::is_idle(self : BuildNextOutcome) -> Bool {
  match self {
    BuildIdle => true
    _ => false
  }
}

///|
pub fn BuildNextOutcome::is_backpressured(self : BuildNextOutcome) -> Bool {
  match self {
    BuildBackpressured => true
    _ => false
  }
}

///|
pub fn BuildNextOutcome::is_submitted(self : BuildNextOutcome) -> Bool {
  match self {
    BuildSubmitted(_, _) => true
    _ => false
  }
}

///|
/// Opaque identity of a scheduler-owned request slot.
#valtype
pub struct SchedulerSlotId(Int, Int) derive(Debug, Eq, Hash)

///|
pub fn SchedulerSlotId::index(self : SchedulerSlotId) -> Int {
  self.0
}

///|
pub fn SchedulerSlotId::generation(self : SchedulerSlotId) -> Int {
  self.1
}

///|
pub(all) enum SchedulerRequestState {
  Waiting
  Prefill
  Decode
  Finishing
  Cancelled
  DeadlineExpired
  Failed
} derive(Debug, Eq)

///|
/// Scheduler-owned terminal reason. Stop-string matching remains outside this
/// slice until a bounded incremental decoder/matcher is integrated.
pub(all) enum TerminalReason {
  CancelledByCaller
  DeadlineReached
  StopTokenReached
  MaximumOutputReached
  WorkerFailed
} derive(Debug, Eq)

///|
/// Slot-free publication identity. It stays valid after request-slot reuse.
#valtype
pub struct TerminalNotice(
  @inference.RequestId,
  @inference.RequestGeneration,
  TerminalReason
) derive(Debug, Eq)

///|
pub fn TerminalNotice::request_id(
  self : TerminalNotice,
) -> @inference.RequestId {
  self.0
}

///|
pub fn TerminalNotice::request_generation(
  self : TerminalNotice,
) -> @inference.RequestGeneration {
  self.1
}

///|
pub fn TerminalNotice::reason(self : TerminalNotice) -> TerminalReason {
  self.2
}

///|
/// Slot-free publication of one generated token. The request identity remains
/// valid after terminal resource release and scheduler-slot reuse.
#valtype
pub struct GeneratedTokenNotice(
  @inference.RequestId,
  @inference.RequestGeneration,
  Int,
  Int
) derive(Eq)

///|
pub fn GeneratedTokenNotice::request_id(
  self : GeneratedTokenNotice,
) -> @inference.RequestId {
  self.0
}

///|
pub fn GeneratedTokenNotice::request_generation(
  self : GeneratedTokenNotice,
) -> @inference.RequestGeneration {
  self.1
}

///|
pub fn GeneratedTokenNotice::token_id(self : GeneratedTokenNotice) -> Int {
  self.2
}

///|
pub fn GeneratedTokenNotice::position(self : GeneratedTokenNotice) -> Int {
  self.3
}

///|
pub struct RequestHandle {
  slot : SchedulerSlotId
  request_id : @inference.RequestId
  request_generation : @inference.RequestGeneration
} derive(Debug, Eq)

///|
pub fn RequestHandle::slot(self : RequestHandle) -> SchedulerSlotId {
  self.slot
}

///|
pub fn RequestHandle::request_id(self : RequestHandle) -> @inference.RequestId {
  self.request_id
}

///|
pub fn RequestHandle::request_generation(
  self : RequestHandle,
) -> @inference.RequestGeneration {
  self.request_generation
}

///|
pub struct RequestStatus {
  handle : RequestHandle
  state : SchedulerRequestState
  input_tokens : Int
  processed_input_tokens : Int
  output_tokens : Int
  deadline_at_millis : UInt64
  in_flight : Bool
} derive(Debug, Eq)

///|
pub fn RequestStatus::handle(self : RequestStatus) -> RequestHandle {
  self.handle
}

///|
pub fn RequestStatus::state(self : RequestStatus) -> SchedulerRequestState {
  self.state
}

///|
pub fn RequestStatus::input_tokens(self : RequestStatus) -> Int {
  self.input_tokens
}

///|
pub fn RequestStatus::processed_input_tokens(self : RequestStatus) -> Int {
  self.processed_input_tokens
}

///|
pub fn RequestStatus::output_tokens(self : RequestStatus) -> Int {
  self.output_tokens
}

///|
pub fn RequestStatus::deadline_at_millis(self : RequestStatus) -> UInt64 {
  self.deadline_at_millis
}

///|
pub fn RequestStatus::in_flight(self : RequestStatus) -> Bool {
  self.in_flight
}

///|
/// Single thread-confined owner of request, physical-page, and block-table
/// metadata. Every mutable collection is fixed at startup.
pub struct Scheduler {
  priv capacity : @runtime_resolved.ResolvedRuntimeCapacity
  priv worker_limits : @worker_protocol.WorkerProtocolLimits
  priv model_identity : @spec.ModelIdentity
  priv page_allocator : @page_allocator.PageAllocator
  priv block_tables : @block_table.BlockTableArena
  priv model_plan_generation : @worker_protocol.ModelPlanGeneration
  priv intermediate_prefill_recipe : @worker_protocol.RowCapabilityRecipe
  priv final_prefill_recipe : @worker_protocol.RowCapabilityRecipe
  priv decode_recipe : @worker_protocol.RowCapabilityRecipe
  priv completion_slots : FixedArray[@worker_protocol.CompletionSlot]
  priv plan_a : @worker_protocol.SchedulePlanBuffer
  priv plan_b : @worker_protocol.SchedulePlanBuffer
  priv completion_a : @worker_protocol.CompletionBuffer
  priv completion_b : @worker_protocol.CompletionBuffer
  priv request_tables : @block_table.BlockTableIdStorage
  priv journal_tables : @block_table.BlockTableIdStorage
  priv page_run : @page_allocator.PageRunBuffer
  priv release_pages : @block_table.BlockTablePageBuffer
  priv slot_states : FixedArray[Int]
  priv slot_generations : FixedArray[Int]
  priv requests : FixedArray[TokenizedRequest?]
  priv request_generations : FixedArray[@inference.RequestGeneration]
  priv deadlines : FixedArray[UInt64]
  priv admitted_steps : FixedArray[UInt64]
  priv processed_inputs : FixedArray[Int]
  priv output_counts : FixedArray[Int]
  priv in_flight : FixedArray[Bool]
  priv decode_tokens : FixedArray[Int]
  priv waiting_next : FixedArray[Int]
  priv free_next : FixedArray[Int]
  priv active_slots : FixedArray[Int]
  priv prefill_slots : FixedArray[Int]
  priv prefill_tokens : FixedArray[Int]
  priv decode_slots : FixedArray[Int]
  priv journal_slots : FixedArray[Int]
  priv journal_old_page_counts : FixedArray[Int]
  priv journal_appended : FixedArray[Bool]
  priv plan_a_slots : FixedArray[Int]
  priv plan_b_slots : FixedArray[Int]
  priv terminal_request_ids : FixedArray[@inference.RequestId]
  priv terminal_generations : FixedArray[@inference.RequestGeneration]
  priv terminal_reasons : FixedArray[TerminalReason]
  priv output_request_ids : FixedArray[@inference.RequestId]
  priv output_generations : FixedArray[@inference.RequestGeneration]
  priv output_token_ids : FixedArray[Int]
  priv output_positions : FixedArray[Int]
  priv mut free_head : Int
  priv mut free_tail : Int
  priv mut waiting_head : Int
  priv mut waiting_tail : Int
  priv mut waiting_count : Int
  priv mut active_count : Int
  priv mut live_count : Int
  priv mut retired_slot_count : Int
  priv mut terminal_head : Int
  priv mut terminal_tail : Int
  priv mut terminal_count : Int
  priv mut output_head : Int
  priv mut output_tail : Int
  priv mut output_count : Int
  priv mut generation_cursor : UInt64
  priv mut step : UInt64
  priv mut last_plan_sequence : UInt64
  priv mut last_retired_plan_sequence : UInt64
  priv mut plan_a_sequence : UInt64
  priv mut plan_b_sequence : UInt64
  priv mut prefill_count : Int
  priv mut decode_count : Int
  priv mut journal_count : Int
}
