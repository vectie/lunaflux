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
  priv model_identity : @spec.ModelIdentity
  priv page_allocator : @page_allocator.PageAllocator
  priv block_tables : @block_table.BlockTableArena
  priv model_plan_generation : @worker_protocol.ModelPlanGeneration
  priv slot_states : FixedArray[Int]
  priv slot_generations : FixedArray[Int]
  priv requests : FixedArray[TokenizedRequest?]
  priv request_generations : FixedArray[@inference.RequestGeneration]
  priv deadlines : FixedArray[UInt64]
  priv admitted_steps : FixedArray[UInt64]
  priv processed_inputs : FixedArray[Int]
  priv output_counts : FixedArray[Int]
  priv in_flight : FixedArray[Bool]
  priv waiting_next : FixedArray[Int]
  priv free_next : FixedArray[Int]
  priv terminal_request_ids : FixedArray[@inference.RequestId]
  priv terminal_generations : FixedArray[@inference.RequestGeneration]
  priv terminal_reasons : FixedArray[TerminalReason]
  priv mut free_head : Int
  priv mut free_tail : Int
  priv mut waiting_head : Int
  priv mut waiting_tail : Int
  priv mut waiting_count : Int
  priv mut live_count : Int
  priv mut retired_slot_count : Int
  priv mut terminal_head : Int
  priv mut terminal_tail : Int
  priv mut terminal_count : Int
  priv mut generation_cursor : UInt64
  priv mut step : UInt64
}
