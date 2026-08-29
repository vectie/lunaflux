///|
pub fn RankGroupOwner::take_failure(
  self : RankGroupOwner,
) -> RankGroupFailureReport raise RankGroupError {
  self.require_lifecycle(Faulted)
  if self.failure_published {
    raise Invalid(FailurePublication)
  }
  self.failure_published = true
  RankGroupFailureReport(
    self.generation.value(),
    self.active_sequence_value,
    self.failure_reason,
    self.failure_rank,
    self.world_size,
  )
}

///|
/// Publishes an exact rank loss observed while no plan is active. A fresh
/// epoch prevents a stale idle observation from sharing authority with any
/// earlier completed plan; the terminal report carries sequence zero.
pub fn RankGroupOwner::report_idle_rank_loss(
  self : RankGroupOwner,
  rank : Int,
) -> Unit raise RankGroupError {
  self.require_lifecycle(Ready)
  validate_rank(rank, self.world_size)
  self.start_epoch()
  self.fault(RankLost, rank, true)
}

///|
/// Authenticates an intended plan when the host cannot establish the exchange
/// clock/deadline. Synthetic internal ticks exist only to bind the exact plan
/// sequence into the one-shot failure report.
pub fn RankGroupOwner::report_begin_host_failure(
  self : RankGroupOwner,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> Unit raise RankGroupError {
  ignore(self.begin(plan, started_tick=0UL, deadline_tick=1UL))
  self.fault(HostFailure, -1, false)
}

///|
pub fn RankGroupOwner::report_active_host_failure(
  self : RankGroupOwner,
) -> Unit raise RankGroupError {
  self.require_lifecycle(Executing)
  self.fault(HostFailure, -1, false)
}

///|
pub fn RankGroupOwner::report_idle_host_failure(
  self : RankGroupOwner,
) -> Unit raise RankGroupError {
  self.require_lifecycle(Ready)
  self.start_epoch()
  self.fault(HostFailure, -1, false)
}

///|
pub fn RankGroupFailureReport::group_generation_value(
  self : RankGroupFailureReport,
) -> UInt64 {
  self.0
}

///|
pub fn RankGroupFailureReport::plan_sequence_value(
  self : RankGroupFailureReport,
) -> UInt64 {
  self.1
}

///|
pub fn RankGroupFailureReport::reason(
  self : RankGroupFailureReport,
) -> RankGroupFailureReason {
  self.2
}

///|
pub fn RankGroupFailureReport::failed_rank(
  self : RankGroupFailureReport,
) -> Int {
  self.3
}

///|
pub fn RankGroupFailureReport::world_size(self : RankGroupFailureReport) -> Int {
  self.4
}

///|
/// True only when the terminal reason carries its canonical failed-rank
/// shape. Host-wide and deadline failures have no singular failed rank;
/// rank-local failures must identify one member of the reported group.
/// Protocol violations may originate either at the group boundary or at one
/// exact rank.
pub fn RankGroupFailureReport::has_canonical_failure_rank(
  self : RankGroupFailureReport,
) -> Bool {
  let rank_is_member = self.3 >= 0 && self.3 < self.4
  match self.2 {
    DeadlineExpired | HostFailure => self.3 == -1
    RankLost | CollectiveFailure | RankExecutionFailure => rank_is_member
    ProtocolViolation(_) => self.3 == -1 || rank_is_member
  }
}

///|
pub fn RankGroupOwner::drain_lease(
  self : RankGroupOwner,
  rank : Int,
) -> RankDrainLease raise RankGroupError {
  self.require_lifecycle(Faulted)
  validate_rank(rank, self.world_size)
  if self.drained_ranks[rank] {
    if self.lost_ranks[rank] {
      raise Invalid(LostRankDrain)
    }
    raise Invalid(DuplicateDrainAck)
  }
  RankDrainLease(self, self.epoch, rank)
}

///|
fn RankDrainLease::authenticate(
  self : RankDrainLease,
) -> RankGroupOwner raise RankGroupError {
  let owner = self.0
  owner.require_lifecycle(Faulted)
  if self.1 != owner.epoch || self.2 < 0 || self.2 >= owner.world_size {
    raise Invalid(Submission)
  }
  if owner.drained_ranks[self.2] {
    if owner.lost_ranks[self.2] {
      raise Invalid(LostRankDrain)
    }
    raise Invalid(DuplicateDrainAck)
  }
  owner
}

///|
pub fn RankDrainLease::rank(self : RankDrainLease) -> Int raise RankGroupError {
  ignore(self.authenticate())
  self.2
}

///|
pub fn RankDrainLease::ack(
  self : RankDrainLease,
) -> RankDrainAck raise RankGroupError {
  let owner = self.authenticate()
  RankDrainAck(
    owner.generation.value(),
    owner.active_sequence_value,
    self.2,
    owner.world_size,
  )
}

///|
pub fn RankDrainAck::from_wire(
  group_generation_value~ : UInt64,
  plan_sequence_value~ : UInt64,
  rank~ : Int,
  world_size~ : Int,
) -> RankDrainAck raise RankGroupError {
  validate_world_size(world_size)
  if group_generation_value == 0UL {
    raise Invalid(GroupGeneration)
  }
  validate_rank(rank, world_size)
  RankDrainAck(group_generation_value, plan_sequence_value, rank, world_size)
}

///|
pub fn RankGroupOwner::accept_drain_ack(
  self : RankGroupOwner,
  ack : RankDrainAck,
) -> Unit raise RankGroupError {
  self.require_lifecycle(Faulted)
  if ack.0 != self.generation.value() {
    raise Invalid(GroupGeneration)
  }
  if ack.1 != self.active_sequence_value {
    raise Invalid(PlanSequence)
  }
  if ack.3 != self.world_size {
    raise Invalid(WorldSize)
  }
  validate_rank(ack.2, self.world_size)
  if self.drained_ranks[ack.2] {
    if self.lost_ranks[ack.2] {
      raise Invalid(LostRankDrain)
    }
    raise Invalid(DuplicateDrainAck)
  }
  self.drained_ranks[ack.2] = true
  self.drain_count += 1
}

///|
/// Excludes one additional rank that is proven lost while a faulted group is
/// draining. The original terminal reason and failed-rank publication remain
/// unchanged; this only prevents recovery from waiting forever for an ACK the
/// dead worker can no longer produce.
pub fn RankGroupOwner::mark_drain_rank_lost(
  self : RankGroupOwner,
  rank : Int,
) -> Unit raise RankGroupError {
  self.require_lifecycle(Faulted)
  validate_rank(rank, self.world_size)
  if self.drained_ranks[rank] {
    if self.lost_ranks[rank] {
      raise Invalid(LostRankDrain)
    }
    raise Invalid(DuplicateDrainAck)
  }
  self.lost_ranks[rank] = true
  self.drained_ranks[rank] = true
  self.drain_count += 1
}

///|
pub fn RankGroupOwner::drain_complete(
  self : RankGroupOwner,
) -> Bool raise RankGroupError {
  self.require_lifecycle(Faulted)
  self.drain_count == self.world_size
}

///|
pub fn RankGroupOwner::close_faulted(
  self : RankGroupOwner,
) -> Unit raise RankGroupError {
  self.require_lifecycle(Faulted)
  if !self.failure_published {
    raise NotReady(FailurePublication)
  }
  if self.drain_count != self.world_size {
    raise NotReady(DrainPending)
  }
  self.lifecycle = Closed
}
