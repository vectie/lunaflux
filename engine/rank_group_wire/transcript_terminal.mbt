///|
fn RankGroupWireTranscript::begin_abort(
  self : RankGroupWireTranscript,
) -> Unit raise RankGroupWireError {
  if self.lifecycle == NewGroup ||
    self.lifecycle == AbortedPlan ||
    self.lifecycle == Draining ||
    self.lifecycle == DrainedGroup ||
    self.lifecycle == Closing ||
    self.lifecycle == ClosedGroup ||
    self.lifecycle == Faulted {
    return self.reject(Lifecycle)
  }
  let mut target_count = 0
  for rank in 0..<self.world_size {
    if self.configured_ranks[rank] {
      self.rank_states[rank] = 10
      target_count += 1
    } else {
      self.rank_states[rank] = -1
    }
  }
  if target_count == 0 {
    return self.reject(Lifecycle)
  }
  self.phase_count = 0
  self.phase_target_count = target_count
  self.lifecycle = Aborting
}

///|
fn RankGroupWireTranscript::accept_abort(
  self : RankGroupWireTranscript,
  sequence : UInt64,
  slot : RankGroupWireSlot?,
  rank : Int,
) -> Unit raise RankGroupWireError {
  if self.lifecycle != Aborting {
    self.begin_abort()
  }
  let expected_sequence = self.active_sequence_value
  if sequence != expected_sequence ||
    wire_slot_code(slot) != self.active_slot_code ||
    self.rank_states[rank] != 10 {
    return self.reject(Sequence)
  }
  self.rank_states[rank] = 11
}

///|
fn RankGroupWireTranscript::accept_aborted(
  self : RankGroupWireTranscript,
  sequence : UInt64,
  slot : RankGroupWireSlot?,
  rank : Int,
) -> Unit raise RankGroupWireError {
  if self.lifecycle != Aborting || self.rank_states[rank] != 11 {
    return self.reject(Lifecycle)
  }
  if sequence != self.active_sequence_value ||
    wire_slot_code(slot) != self.active_slot_code {
    return self.reject(Sequence)
  }
  self.rank_states[rank] = 12
  self.phase_count += 1
  if self.phase_count == self.phase_target_count {
    if self.active_sequence_value != 0UL {
      self.previous_sequence_value = self.active_sequence_value
    }
    self.active_sequence_value = 0UL
    self.active_slot_code = -1
    self.lifecycle = AbortedPlan
  }
}

///|
fn RankGroupWireTranscript::begin_terminal_phase(
  self : RankGroupWireTranscript,
  state : Int,
  lifecycle : RankGroupWireLifecycle,
) -> Unit {
  let mut target_count = 0
  for rank in 0..<self.world_size {
    if self.configured_ranks[rank] {
      self.rank_states[rank] = state
      target_count += 1
    } else {
      self.rank_states[rank] = -1
    }
  }
  self.phase_count = 0
  self.phase_target_count = target_count
  self.lifecycle = lifecycle
}
