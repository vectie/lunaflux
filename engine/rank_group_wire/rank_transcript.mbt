///|
/// Authority-free transcript for exactly one child rank. It authenticates
/// every post-Configure frame against the startup-locked binding before any
/// lifecycle state is changed.
pub struct RankGroupWireRankTranscript {
  priv mut binding_value : RankGroupWireBinding?
  priv mut lifecycle_value : RankGroupWireLifecycle
  priv mut previous_sequence : UInt64
  priv mut active_sequence : UInt64
  priv mut active_slot_code : Int
  priv mut startup_admitted : Bool
  priv mut fault_rule_value : RankGroupWireRule
  priv mut plan_failure_value : RankGroupWirePlanFailure?
}

///|
pub fn RankGroupWireRankTranscript::new() -> RankGroupWireRankTranscript {
  {
    binding_value: None,
    lifecycle_value: NewGroup,
    previous_sequence: 0UL,
    active_sequence: 0UL,
    active_slot_code: -1,
    startup_admitted: false,
    fault_rule_value: Lifecycle,
    plan_failure_value: None,
  }
}

///|
pub fn RankGroupWireRankTranscript::lifecycle(
  self : RankGroupWireRankTranscript,
) -> RankGroupWireLifecycle {
  self.lifecycle_value
}

///|
pub fn RankGroupWireRankTranscript::binding(
  self : RankGroupWireRankTranscript,
) -> RankGroupWireBinding? {
  self.binding_value
}

///|
pub fn RankGroupWireRankTranscript::previous_sequence_value(
  self : RankGroupWireRankTranscript,
) -> UInt64 {
  self.previous_sequence
}

///|
pub fn RankGroupWireRankTranscript::active_sequence_value(
  self : RankGroupWireRankTranscript,
) -> UInt64 {
  self.active_sequence
}

///|
pub fn RankGroupWireRankTranscript::active_slot(
  self : RankGroupWireRankTranscript,
) -> RankGroupWireSlot? {
  match self.active_slot_code {
    0 => Some(A)
    1 => Some(B)
    _ => None
  }
}

///|
pub fn RankGroupWireRankTranscript::failure_rule(
  self : RankGroupWireRankTranscript,
) -> RankGroupWireRule? {
  if self.lifecycle_value == Faulted {
    Some(self.fault_rule_value)
  } else {
    None
  }
}

///|
/// Returns the exact locally reported active-plan failure, when present.
pub fn RankGroupWireRankTranscript::plan_failure(
  self : RankGroupWireRankTranscript,
) -> RankGroupWirePlanFailure? {
  self.plan_failure_value
}

///|
fn RankGroupWireRankTranscript::poison_rank(
  self : RankGroupWireRankTranscript,
  rule : RankGroupWireRule,
) -> Unit {
  if self.lifecycle_value != Faulted {
    self.lifecycle_value = Faulted
    self.fault_rule_value = rule
  }
}

///|
fn[T] RankGroupWireRankTranscript::reject_rank(
  self : RankGroupWireRankTranscript,
  rule : RankGroupWireRule,
) -> T raise RankGroupWireError {
  if self.lifecycle_value == Faulted {
    raise TranscriptFaulted(self.fault_rule_value)
  }
  self.poison_rank(rule)
  raise ProtocolFault(rule)
}

///|
/// Decodes and locks the first Configure binding. The opaque Configure payload
/// remains in caller-owned storage and is not retained by this transcript.
pub fn RankGroupWireRankTranscript::accept_configure(
  self : RankGroupWireRankTranscript,
  source : FixedArray[Byte],
  length : Int,
  limits : RankGroupWireLimits,
) -> RankGroupWireDecodedConfigure raise RankGroupWireError {
  if self.lifecycle_value == Faulted {
    raise TranscriptFaulted(self.fault_rule_value)
  }
  if self.lifecycle_value != NewGroup || self.binding_value is Some(_) {
    return self.reject_rank(Lifecycle)
  }
  let decoded = decode_rank_group_wire_configure(source, length, limits) catch {
    error => return self.reject_rank(error_rule(error))
  }
  self.binding_value = Some(decoded.binding())
  self.lifecycle_value = Configuring
  decoded
}

///|
/// Binds the locally admitted startup identity and exact predecessor before a
/// Ready response can be accepted. It performs no artifact or device I/O.
pub fn RankGroupWireRankTranscript::admit_ready(
  self : RankGroupWireRankTranscript,
  admitted_binding : RankGroupWireBinding,
  previous_sequence_value~ : UInt64,
) -> Unit raise RankGroupWireError {
  if self.lifecycle_value == Faulted {
    raise TranscriptFaulted(self.fault_rule_value)
  }
  let locked = match self.binding_value {
    Some(binding) => binding
    None => return self.reject_rank(Binding)
  }
  if self.lifecycle_value != Configuring || self.startup_admitted {
    return self.reject_rank(Lifecycle)
  }
  if previous_sequence_value == 0xffffffffffffffffUL {
    return self.reject_rank(Sequence)
  }
  if !locked.exactly_matches(admitted_binding) {
    return self.reject_rank(Binding)
  }
  self.previous_sequence = previous_sequence_value
  self.startup_admitted = true
}

///|
fn RankGroupWireRankTranscript::require_active_rank_shape(
  self : RankGroupWireRankTranscript,
  header : RankGroupWireDecodedHeader,
) -> Unit raise RankGroupWireError {
  if header.sequence_value() != self.active_sequence ||
    wire_slot_code(header.slot()) != self.active_slot_code {
    return self.reject_rank(Sequence)
  }
}

///|
fn RankGroupWireRankTranscript::accept_rank_header(
  self : RankGroupWireRankTranscript,
  header : RankGroupWireDecodedHeader,
) -> Unit raise RankGroupWireError {
  match header.kind() {
    Configure => return self.reject_rank(Lifecycle)
    Ready => {
      if self.lifecycle_value != Configuring || !self.startup_admitted {
        return self.reject_rank(Lifecycle)
      }
      self.lifecycle_value = ReadyGroup
    }
    Submit => {
      let sequence = header.sequence_value()
      if self.lifecycle_value != ReadyGroup ||
        self.previous_sequence == 0xffffffffffffffffUL ||
        sequence != self.previous_sequence + 1UL {
        return self.reject_rank(Sequence)
      }
      self.active_sequence = sequence
      self.active_slot_code = wire_slot_code(header.slot())
      self.plan_failure_value = None
      self.lifecycle_value = Executing
    }
    PollComplete => {
      if self.lifecycle_value != Executing {
        return self.reject_rank(Lifecycle)
      }
      self.require_active_rank_shape(header)
      self.lifecycle_value = Polling
    }
    Pending => {
      if self.lifecycle_value != Polling {
        return self.reject_rank(Lifecycle)
      }
      self.require_active_rank_shape(header)
      self.lifecycle_value = Executing
    }
    Complete => {
      if self.lifecycle_value != Polling {
        return self.reject_rank(Lifecycle)
      }
      self.require_active_rank_shape(header)
      let locked = match self.binding_value {
        Some(binding) => binding
        None => return self.reject_rank(Binding)
      }
      if (locked.rank() == 0 && header.payload_length() == 0) ||
        (locked.rank() != 0 && header.payload_length() != 0) {
        return self.reject_rank(Payload)
      }
      self.previous_sequence = self.active_sequence
      self.active_sequence = 0UL
      self.active_slot_code = -1
      self.lifecycle_value = ReadyGroup
    }
    PlanFailed(failure) => {
      if self.lifecycle_value != Polling {
        return self.reject_rank(Lifecycle)
      }
      self.require_active_rank_shape(header)
      self.previous_sequence = self.active_sequence
      self.active_sequence = 0UL
      self.active_slot_code = -1
      self.plan_failure_value = Some(failure)
      self.lifecycle_value = AbortedPlan
    }
    Abort => {
      if self.lifecycle_value == NewGroup ||
        self.lifecycle_value == Aborting ||
        self.lifecycle_value == AbortedPlan ||
        self.lifecycle_value == ClosedGroup ||
        self.lifecycle_value == Draining ||
        self.lifecycle_value == DrainedGroup ||
        self.lifecycle_value == Closing {
        return self.reject_rank(Lifecycle)
      }
      if header.sequence_value() != self.active_sequence ||
        wire_slot_code(header.slot()) != self.active_slot_code {
        return self.reject_rank(Sequence)
      }
      self.lifecycle_value = Aborting
    }
    Aborted => {
      if self.lifecycle_value != Aborting {
        return self.reject_rank(Lifecycle)
      }
      self.require_active_rank_shape(header)
      if self.active_sequence != 0UL {
        self.previous_sequence = self.active_sequence
      }
      self.active_sequence = 0UL
      self.active_slot_code = -1
      self.lifecycle_value = AbortedPlan
    }
    Drain => {
      if self.lifecycle_value == Executing || self.lifecycle_value == Polling {
        self.previous_sequence = self.active_sequence
        self.active_sequence = 0UL
        self.active_slot_code = -1
      } else if self.lifecycle_value != ReadyGroup &&
        self.lifecycle_value != AbortedPlan {
        return self.reject_rank(Lifecycle)
      }
      self.lifecycle_value = Draining
    }
    Drained => {
      if self.lifecycle_value != Draining {
        return self.reject_rank(Lifecycle)
      }
      self.lifecycle_value = DrainedGroup
    }
    Close => {
      if self.lifecycle_value != DrainedGroup {
        return self.reject_rank(Lifecycle)
      }
      self.lifecycle_value = Closing
    }
    Closed => {
      if self.lifecycle_value != Closing {
        return self.reject_rank(Lifecycle)
      }
      self.lifecycle_value = ClosedGroup
    }
  }
}

///|
/// Validates one subsequent frame against the locked binding and then advances
/// the rank-local transcript. The returned value contains copied scalars only.
fn RankGroupWireRankTranscript::accept_direction(
  self : RankGroupWireRankTranscript,
  source : FixedArray[Byte],
  length : Int,
  limits : RankGroupWireLimits,
  expected : RankGroupWireDirection,
) -> RankGroupWireDecodedHeader raise RankGroupWireError {
  if self.lifecycle_value == Faulted {
    raise TranscriptFaulted(self.fault_rule_value)
  }
  if self.lifecycle_value == ClosedGroup {
    return self.reject_rank(Lifecycle)
  }
  let locked = match self.binding_value {
    Some(binding) => binding
    None => return self.reject_rank(Binding)
  }
  let header = validate_rank_group_wire_binding(source, length, limits, locked) catch {
    error => return self.reject_rank(error_rule(error))
  }
  if header.direction() != expected {
    return self.reject_rank(Direction)
  }
  self.accept_rank_header(header)
  header
}

///|
/// Authenticates and accepts one parent-to-child command.
pub fn RankGroupWireRankTranscript::accept_parent(
  self : RankGroupWireRankTranscript,
  source : FixedArray[Byte],
  length : Int,
  limits : RankGroupWireLimits,
) -> RankGroupWireDecodedHeader raise RankGroupWireError {
  self.accept_direction(source, length, limits, ParentToChild)
}

///|
/// Authenticates and accepts one child-to-parent response.
pub fn RankGroupWireRankTranscript::accept_child(
  self : RankGroupWireRankTranscript,
  source : FixedArray[Byte],
  length : Int,
  limits : RankGroupWireLimits,
) -> RankGroupWireDecodedHeader raise RankGroupWireError {
  self.accept_direction(source, length, limits, ChildToParent)
}

///|
/// Irrevocably faults the local transcript after transport loss. No ACK is
/// synthesized and no unavailable peer is awaited.
pub fn RankGroupWireRankTranscript::abandon_on_transport_loss(
  self : RankGroupWireRankTranscript,
) -> Unit {
  if self.lifecycle_value != ClosedGroup && self.lifecycle_value != Faulted {
    self.poison_rank(TransportLost)
  }
}
