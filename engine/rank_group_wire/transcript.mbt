///|
fn error_rule(error : RankGroupWireError) -> RankGroupWireRule {
  match error {
    Invalid(rule) | ProtocolFault(rule) | TranscriptFaulted(rule) => rule
  }
}

///|
fn RankGroupWireTranscript::poison(
  self : RankGroupWireTranscript,
  rule : RankGroupWireRule,
) -> Unit {
  if self.lifecycle != Faulted {
    self.lifecycle = Faulted
    self.fault_rule = rule
  }
}

///|
fn[T] RankGroupWireTranscript::reject(
  self : RankGroupWireTranscript,
  rule : RankGroupWireRule,
) -> T raise RankGroupWireError {
  if self.lifecycle == Faulted {
    raise TranscriptFaulted(self.fault_rule)
  }
  self.poison(rule)
  raise ProtocolFault(rule)
}

///|
fn copy_group_bindings(
  source : ArrayView[RankGroupWireBinding],
) -> FixedArray[RankGroupWireBinding] raise RankGroupWireError {
  if source.length() < 2 || source.length() > RANK_GROUP_WIRE_MAX_WORLD_SIZE {
    raise Invalid(WorldSize)
  }
  let world_size = source.length()
  let first = source[0]
  if first.world_size != world_size || first.rank != 0 {
    raise Invalid(Binding)
  }
  let copied = FixedArray::make(world_size, first)
  for rank in 0..<world_size {
    let binding = source[rank]
    if binding.rank != rank ||
      binding.world_size != world_size ||
      !binding.same_group_as(first) {
      raise Invalid(Binding)
    }
    copied[rank] = binding
  }
  copied
}

///|
pub fn RankGroupWireTranscript::new(
  bindings : ArrayView[RankGroupWireBinding],
  previous_sequence_value~ : UInt64,
) -> RankGroupWireTranscript raise RankGroupWireError {
  if previous_sequence_value == 0xffffffffffffffffUL {
    raise Invalid(Sequence)
  }
  let copied = copy_group_bindings(bindings)
  let world_size = copied.length()
  {
    bindings: copied,
    rank_states: FixedArray::make(world_size, 0),
    configured_ranks: FixedArray::make(world_size, false),
    world_size,
    lifecycle: NewGroup,
    previous_sequence_value,
    active_sequence_value: 0UL,
    active_slot_code: -1,
    phase_count: 0,
    phase_target_count: 0,
    fault_rule: Lifecycle,
    plan_failure_value: None,
    plan_failure_rank_value: -1,
  }
}

///|
pub fn RankGroupWireTranscript::lifecycle(
  self : RankGroupWireTranscript,
) -> RankGroupWireLifecycle {
  self.lifecycle
}

///|
pub fn RankGroupWireTranscript::world_size(
  self : RankGroupWireTranscript,
) -> Int {
  self.world_size
}

///|
pub fn RankGroupWireTranscript::previous_sequence_value(
  self : RankGroupWireTranscript,
) -> UInt64 {
  self.previous_sequence_value
}

///|
pub fn RankGroupWireTranscript::active_sequence_value(
  self : RankGroupWireTranscript,
) -> UInt64 {
  self.active_sequence_value
}

///|
pub fn RankGroupWireTranscript::active_slot(
  self : RankGroupWireTranscript,
) -> RankGroupWireSlot? {
  match self.active_slot_code {
    0 => Some(A)
    1 => Some(B)
    _ => None
  }
}

///|
pub fn RankGroupWireTranscript::failure_rule(
  self : RankGroupWireTranscript,
) -> RankGroupWireRule? {
  if self.lifecycle == Faulted {
    Some(self.fault_rule)
  } else {
    None
  }
}

///|
/// Returns the exact authenticated active-plan failure, when one retired the
/// current sequence without a successful completion.
pub fn RankGroupWireTranscript::plan_failure(
  self : RankGroupWireTranscript,
) -> RankGroupWirePlanFailure? {
  self.plan_failure_value
}

///|
/// Returns the reporting rank for the authenticated active-plan failure.
pub fn RankGroupWireTranscript::plan_failure_rank(
  self : RankGroupWireTranscript,
) -> Int? {
  if self.plan_failure_value is Some(_) {
    Some(self.plan_failure_rank_value)
  } else {
    None
  }
}

///|
/// Irrevocably abandons this authority-free transcript after transport EOF or
/// child-process loss. The supervisor separately reports the exact lost-rank
/// event to `rank_group_protocol`; no unavailable child ACK is awaited here.
pub fn RankGroupWireTranscript::abandon_on_transport_loss(
  self : RankGroupWireTranscript,
) -> Unit {
  if self.lifecycle != ClosedGroup && self.lifecycle != Faulted {
    self.poison(TransportLost)
  }
}

///|
pub fn RankGroupWireTranscript::binding(
  self : RankGroupWireTranscript,
  rank : Int,
) -> RankGroupWireBinding? {
  if rank < 0 || rank >= self.world_size {
    None
  } else {
    Some(self.bindings[rank])
  }
}

///|
fn RankGroupWireTranscript::authenticate_frame(
  self : RankGroupWireTranscript,
  frame : ValidatedRankGroupWireFrame,
) -> Int raise RankGroupWireError {
  let owner = frame.owner() catch {
    error => return self.reject(error_rule(error))
  }
  let rank = wire_read_i32(owner.storage, 24)
  if rank < 0 || rank >= self.world_size {
    return self.reject(Rank)
  }
  let matches = frame.matches_binding(self.bindings[rank]) catch {
    error => return self.reject(error_rule(error))
  }
  if !matches {
    return self.reject(Binding)
  }
  rank
}

///|
fn RankGroupWireTranscript::clear_rank_states(
  self : RankGroupWireTranscript,
) -> Unit {
  for rank in 0..<self.world_size {
    self.rank_states[rank] = 0
  }
  self.phase_count = 0
  self.phase_target_count = 0
}

///|
fn RankGroupWireTranscript::has_pending_poll(
  self : RankGroupWireTranscript,
) -> Bool {
  for state in self.rank_states {
    if state == 2 {
      return true
    }
  }
  false
}

///|
fn RankGroupWireTranscript::all_ranks_submitted(
  self : RankGroupWireTranscript,
) -> Bool {
  for rank in 0..<self.world_size {
    if self.rank_states[rank] == 0 {
      return false
    }
  }
  true
}

///|
fn RankGroupWireTranscript::require_active_shape(
  self : RankGroupWireTranscript,
  sequence : UInt64,
  slot : RankGroupWireSlot?,
) -> Unit raise RankGroupWireError {
  if sequence != self.active_sequence_value ||
    wire_slot_code(slot) != self.active_slot_code {
    return self.reject(Sequence)
  }
}

///|
fn RankGroupWireTranscript::accept_rank_header(
  self : RankGroupWireTranscript,
  header : RankGroupWireDecodedHeader,
  rank : Int,
) -> Unit raise RankGroupWireError {
  let kind = header.kind()
  let sequence = header.sequence_value()
  let slot = header.slot()
  match kind {
    Configure => {
      if self.lifecycle == NewGroup {
        self.lifecycle = Configuring
      }
      if self.lifecycle != Configuring || self.rank_states[rank] != 0 {
        return self.reject(Lifecycle)
      }
      self.rank_states[rank] = 1
      self.configured_ranks[rank] = true
      self.phase_count += 1
    }
    Ready => {
      if self.lifecycle != Configuring ||
        self.phase_count != self.world_size ||
        self.rank_states[rank] != 1 {
        return self.reject(Lifecycle)
      }
      self.rank_states[rank] = 2
      self.phase_target_count += 1
      if self.phase_target_count == self.world_size {
        self.clear_rank_states()
        self.lifecycle = ReadyGroup
      }
    }
    Submit => {
      if self.lifecycle == ReadyGroup {
        if self.previous_sequence_value == 0xffffffffffffffffUL ||
          sequence != self.previous_sequence_value + 1UL {
          return self.reject(Sequence)
        }
        self.clear_rank_states()
        self.active_sequence_value = sequence
        self.active_slot_code = wire_slot_code(slot)
        self.plan_failure_value = None
        self.plan_failure_rank_value = -1
        self.lifecycle = Executing
      }
      if self.lifecycle != Executing ||
        sequence != self.active_sequence_value ||
        wire_slot_code(slot) != self.active_slot_code ||
        self.rank_states[rank] != 0 {
        return self.reject(Sequence)
      }
      self.rank_states[rank] = 1
      self.phase_count += 1
    }
    PollComplete => {
      if (self.lifecycle != Executing && self.lifecycle != Polling) ||
        !self.all_ranks_submitted() ||
        self.rank_states[rank] != 1 {
        return self.reject(Lifecycle)
      }
      self.require_active_shape(sequence, slot)
      self.rank_states[rank] = 2
      self.lifecycle = Polling
    }
    Pending => {
      if self.lifecycle != Polling || self.rank_states[rank] != 2 {
        return self.reject(Lifecycle)
      }
      self.require_active_shape(sequence, slot)
      self.rank_states[rank] = 1
      if !self.has_pending_poll() {
        self.lifecycle = Executing
      }
    }
    Complete => {
      if self.lifecycle != Polling || self.rank_states[rank] != 2 {
        return self.reject(Lifecycle)
      }
      self.require_active_shape(sequence, slot)
      let payload_length = header.payload_length()
      if (rank == 0 && payload_length == 0) ||
        (rank != 0 && payload_length != 0) {
        return self.reject(Payload)
      }
      self.rank_states[rank] = 3
      self.phase_count -= 1
      if self.phase_count == 0 {
        self.previous_sequence_value = self.active_sequence_value
        self.active_sequence_value = 0UL
        self.active_slot_code = -1
        self.clear_rank_states()
        self.lifecycle = ReadyGroup
      } else if !self.has_pending_poll() {
        self.lifecycle = Executing
      }
    }
    PlanFailed(failure) => {
      if self.lifecycle != Polling || self.rank_states[rank] != 2 {
        return self.reject(Lifecycle)
      }
      self.require_active_shape(sequence, slot)
      self.previous_sequence_value = self.active_sequence_value
      self.active_sequence_value = 0UL
      self.active_slot_code = -1
      self.plan_failure_value = Some(failure)
      self.plan_failure_rank_value = rank
      self.clear_rank_states()
      self.lifecycle = AbortedPlan
    }
    Abort => self.accept_abort(sequence, slot, rank)
    Aborted => self.accept_aborted(sequence, slot, rank)
    Drain => {
      if self.lifecycle == ReadyGroup || self.lifecycle == AbortedPlan {
        self.begin_terminal_phase(20, Draining)
      }
      if self.lifecycle != Draining || self.rank_states[rank] != 20 {
        return self.reject(Lifecycle)
      }
      self.rank_states[rank] = 21
    }
    Drained => {
      if self.lifecycle != Draining || self.rank_states[rank] != 21 {
        return self.reject(Lifecycle)
      }
      self.rank_states[rank] = 22
      self.phase_count += 1
      if self.phase_count == self.phase_target_count {
        self.lifecycle = DrainedGroup
      }
    }
    Close => {
      if self.lifecycle == DrainedGroup {
        self.begin_terminal_phase(30, Closing)
      }
      if self.lifecycle != Closing || self.rank_states[rank] != 30 {
        return self.reject(Lifecycle)
      }
      self.rank_states[rank] = 31
    }
    Closed => {
      if self.lifecycle != Closing || self.rank_states[rank] != 31 {
        return self.reject(Lifecycle)
      }
      self.rank_states[rank] = 32
      self.phase_count += 1
      if self.phase_count == self.phase_target_count {
        self.lifecycle = ClosedGroup
      }
    }
  }
}

///|
pub fn RankGroupWireTranscript::accept(
  self : RankGroupWireTranscript,
  frame : ValidatedRankGroupWireFrame,
) -> Unit raise RankGroupWireError {
  if self.lifecycle == Faulted {
    raise TranscriptFaulted(self.fault_rule)
  }
  if self.lifecycle == ClosedGroup {
    return self.reject(Lifecycle)
  }
  let rank = self.authenticate_frame(frame)
  let header = RankGroupWireDecodedHeader::{
    length: frame.length() catch {
      error => return self.reject(error_rule(error))
    },
    kind: frame.kind() catch {
      error => return self.reject(error_rule(error))
    },
    direction: frame.direction() catch {
      error => return self.reject(error_rule(error))
    },
    slot: frame.slot() catch {
      error => return self.reject(error_rule(error))
    },
    sequence_value: frame.sequence_value() catch {
      error => return self.reject(error_rule(error))
    },
    payload_length: frame.payload_length() catch {
      error => return self.reject(error_rule(error))
    },
  }
  self.accept_rank_header(header, rank)
}

///|
/// Validates and advances one frame directly from caller-owned fixed storage.
/// No frame-owner arena is allocated or retained by the transcript.
pub fn RankGroupWireTranscript::accept_bytes(
  self : RankGroupWireTranscript,
  source : FixedArray[Byte],
  length : Int,
  limits : RankGroupWireLimits,
) -> RankGroupWireDecodedHeader raise RankGroupWireError {
  if self.lifecycle == Faulted {
    raise TranscriptFaulted(self.fault_rule)
  }
  if self.lifecycle == ClosedGroup {
    return self.reject(Lifecycle)
  }
  if length < RANK_GROUP_WIRE_HEADER_BYTES || length > source.length() {
    return self.reject(Length)
  }
  let rank = wire_read_i32(source, 24)
  if rank < 0 || rank >= self.world_size {
    return self.reject(Rank)
  }
  let header = validate_rank_group_wire_binding(
    source,
    length,
    limits,
    self.bindings[rank],
  ) catch {
    error => return self.reject(error_rule(error))
  }
  self.accept_rank_header(header, rank)
  header
}

///|
pub fn RankGroupWireTranscript::load_and_accept(
  self : RankGroupWireTranscript,
  receiver : RankGroupWireFrameBuffer,
  source : FixedArray[Byte],
  length : Int,
) -> Unit raise RankGroupWireError {
  let frame = receiver.load(source, length) catch {
    error => {
      self.poison(error_rule(error))
      raise ProtocolFault(error_rule(error))
    }
  }
  self.accept(frame)
}
