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
    streaming_rank: -1,
    streaming_sequence: 0UL,
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
) -> Int32 {
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
) -> Int32? {
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
  rank : Int32,
) -> RankGroupWireBinding? {
  if rank < 0 || rank >= self.world_size {
    None
  } else {
    Some(self.bindings[rank])
  }
}
