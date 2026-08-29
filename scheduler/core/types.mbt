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
  BuildPreempted
  BuildSubmitted(Int, UInt64)
}

///|
/// Allocation-free result of one authenticated completion attempt. Capacity
/// pressure is ordinary scheduler flow; semantic and ownership failures remain
/// typed `SchedulerError` values.
#valtype
pub(all) enum CompletionCommit {
  CompletionCommitted
  CompletionBackpressured
} derive(Debug, Eq)

///|
/// Allocation-free progress for deterministic instance-loss retirement.
#valtype
pub(all) enum InstanceLossDrain {
  Remaining(Int)
  Complete
} derive(Debug, Eq)

///|
/// Allocation-free result of exact cancellation preflight. A naturally
/// terminal request has already released its scheduler slot and must not be
/// cancelled into a new generation.
#valtype
pub(all) enum CancelReservationStart {
  CancelReserved
  CancelNaturalTerminal
} derive(Debug, Eq)

///|
/// Non-raising result of committing the owner-resident reservation.
#valtype
pub(all) enum CancelReservationCommit {
  CancelCommitted(@inference.RequestGeneration)
  CancelReservationAbsent
} derive(Debug, Eq)

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
/// Opaque caller-owned route copied into every scheduler publication. The
/// scalar route carries no scheduler mutation authority; binding remains
/// authenticated by the exact prepared admission owner.
#valtype
pub struct SchedulerPublicationRoute {
  priv lane_value : Int
  priv generation_value : UInt64
} derive(Eq)

///|
fn empty_publication_route() -> SchedulerPublicationRoute {
  { lane_value: -1, generation_value: 0UL }
}

///|
pub fn SchedulerPublicationRoute::new(
  lane : Int,
  generation : UInt64,
) -> SchedulerPublicationRoute raise SchedulerError {
  if lane < 0 || generation == 0UL || generation > 0xffffffffUL {
    raise invalid(Request, OutOfRange)
  }
  { lane_value: lane, generation_value: generation }
}

///|
pub fn SchedulerPublicationRoute::lane(self : SchedulerPublicationRoute) -> Int {
  self.lane_value
}

///|
pub fn SchedulerPublicationRoute::generation(
  self : SchedulerPublicationRoute,
) -> UInt64 {
  self.generation_value
}

///|
pub(all) enum SchedulerRequestState {
  Waiting
  RecomputeWaiting
  Prefill
  RecomputePrefill
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
/// Identifies which fixed publication ring owns the globally oldest notice.
#valtype
pub(all) enum SchedulerPublicationKind {
  GeneratedTokenPublication
  TerminalPublication
} derive(Debug, Eq)

///|
/// Slot-free publication identity. It stays valid after request-slot reuse.
#valtype
pub struct TerminalNotice(
  @inference.RequestId,
  @inference.RequestGeneration,
  TerminalReason,
  Int,
  UInt64
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
/// Returns the exact full-page prompt tokens reused by this request. The
/// scheduler snapshots this scalar before releasing its prefix reference and
/// recycling the request slot.
pub fn TerminalNotice::cached_input_tokens(self : TerminalNotice) -> Int {
  self.3
}

///|
pub fn TerminalNotice::publication_sequence(self : TerminalNotice) -> UInt64 {
  self.4
}

///|
/// Slot-free publication of one generated token. The request identity remains
/// valid after terminal resource release and scheduler-slot reuse.
#valtype
pub struct GeneratedTokenNotice(
  @inference.RequestId,
  @inference.RequestGeneration,
  Int,
  Int,
  UInt64
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
pub fn GeneratedTokenNotice::publication_sequence(
  self : GeneratedTokenNotice,
) -> UInt64 {
  self.4
}

///|
/// One globally ordered scheduler publication. Consumers should prefer this
/// owner-selected value over choosing between the two physical rings. Its
/// representation is private so external code cannot forge scheduler output.
#valtype
pub struct SchedulerPublication(
  UInt64,
  @inference.RequestId,
  @inference.RequestGeneration,
  UInt64,
  UInt64,
  UInt64
) derive(Eq)

///|
fn scheduler_publication_owner(
  kind : Int,
  slot_index : Int,
  slot_generation : Int,
) -> UInt64 {
  (kind.to_uint64() << 63) |
  (slot_generation.to_uint64() << 32) |
  slot_index.to_uint64()
}

///|
fn terminal_reason_code(reason : TerminalReason) -> Int {
  match reason {
    CancelledByCaller => 0
    DeadlineReached => 1
    StopTokenReached => 2
    MaximumOutputReached => 3
    WorkerFailed => 4
  }
}

///|
fn terminal_reason_from_code(code : Int) -> TerminalReason {
  match code {
    0 => CancelledByCaller
    1 => DeadlineReached
    2 => StopTokenReached
    3 => MaximumOutputReached
    4 => WorkerFailed
    _ => abort("scheduler publication reason invariant")
  }
}

///|
pub fn SchedulerPublication::kind(
  self : SchedulerPublication,
) -> SchedulerPublicationKind {
  if self.0 >> 63 == 0UL {
    GeneratedTokenPublication
  } else {
    TerminalPublication
  }
}

///|
pub fn SchedulerPublication::request_id(
  self : SchedulerPublication,
) -> @inference.RequestId {
  self.1
}

///|
pub fn SchedulerPublication::request_generation(
  self : SchedulerPublication,
) -> @inference.RequestGeneration {
  self.2
}

///|
pub fn SchedulerPublication::publication_sequence(
  self : SchedulerPublication,
) -> UInt64 {
  self.4
}

///|
/// Returns the exact scheduler storage generation that owned this publication.
/// Service owners use it only to route a dequeued publication to private
/// request lanes; it grants no scheduler mutation authority.
pub fn SchedulerPublication::request_slot(
  self : SchedulerPublication,
) -> SchedulerSlotId {
  SchedulerSlotId(
    (self.0 & 0xffffffffUL).to_int(),
    ((self.0 >> 32) & 0x7fffffffUL).to_int(),
  )
}

///|
/// Returns the exact caller route that was bound before scheduler commit.
pub fn SchedulerPublication::route(
  self : SchedulerPublication,
) -> SchedulerPublicationRoute {
  {
    lane_value: (self.5 & 0xffffffffUL).to_int(),
    generation_value: self.5 >> 32,
  }
}

///|
pub fn SchedulerPublication::generated_token_notice(
  self : SchedulerPublication,
) -> GeneratedTokenNotice raise SchedulerError {
  if self.0 >> 63 != 0UL {
    raise invalid(PublicationSequence, OutOfOrder)
  }
  GeneratedTokenNotice(
    self.1,
    self.2,
    (self.3 & 0xffffffffUL).to_int(),
    (self.3 >> 32).to_int(),
    self.4,
  )
}

///|
pub fn SchedulerPublication::terminal_notice(
  self : SchedulerPublication,
) -> TerminalNotice raise SchedulerError {
  if self.0 >> 63 != 1UL {
    raise invalid(PublicationSequence, OutOfOrder)
  }
  TerminalNotice(
    self.1,
    self.2,
    terminal_reason_from_code((self.3 & 0xffffffffUL).to_int()),
    (self.3 >> 32).to_int(),
    self.4,
  )
}

///|
pub struct RequestHandle {
  mut slot : SchedulerSlotId
  request_id : @inference.RequestId
  mut request_generation : @inference.RequestGeneration
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
/// Immutable scalar snapshot of one scheduler owner.
pub struct LunaSchedulerTelemetry {
  priv queue_depth : Int
  priv active_requests : Int
  priv kv_pages_used : Int
  priv kv_pages_free : Int
  priv prefix_lookups : UInt64
  priv prefix_hits : UInt64
  priv prefix_misses : UInt64
  priv prefix_evictions : UInt64
  priv prefix_tokens_reused : UInt64
  priv prefix_tokens_computed : UInt64
  priv prefix_publications : UInt64
  priv prefix_entries : Int
  priv prefix_pages : Int
}

///|
#valtype
pub struct LunaSchedulerPlanTelemetry {
  priv plan_sequence : UInt64
  priv plan_rows : Int
  priv plan_tokens : Int
}

///|
/// Immutable construction evidence for one scheduler owner. Reusing a
/// blueprint constructs independent schedulers; it never aliases mutable
/// request, KV, plan, completion, or publication state.
pub struct SchedulerBlueprint {
  priv capacity : @runtime_resolved.ResolvedRuntimeCapacity
  priv model_identity : @spec.ModelIdentity
  priv model_plan_generation : @worker_protocol.ModelPlanGeneration
  priv intermediate_prefill_recipe : @worker_protocol.RowCapabilityRecipe
  priv final_prefill_recipe : @worker_protocol.RowCapabilityRecipe
  priv decode_recipe : @worker_protocol.RowCapabilityRecipe
  priv worker_limits : @worker_protocol.WorkerProtocolLimits
}
