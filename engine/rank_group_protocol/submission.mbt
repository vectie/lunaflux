///|
fn RankGroupOwner::fault(
  self : RankGroupOwner,
  reason : RankGroupFailureReason,
  failed_rank : Int,
  rank_is_lost : Bool,
) -> Unit {
  self.lifecycle = Faulted
  self.failure_reason = reason
  self.failure_rank = failed_rank
  self.failure_published = false
  self.drain_count = 0
  for rank in 0..<self.world_size {
    self.drained_ranks[rank] = false
    self.lost_ranks[rank] = false
  }
  if rank_is_lost && failed_rank >= 0 && failed_rank < self.world_size {
    self.drained_ranks[failed_rank] = true
    self.lost_ranks[failed_rank] = true
    self.drain_count = 1
  }
}

///|
fn RankGroupOwner::protocol_fault(
  self : RankGroupOwner,
  rule : RankGroupRule,
  rank : Int,
) -> Unit {
  self.fault(ProtocolViolation(rule), rank, false)
}

///|
fn[T] RankGroupOwner::reject_protocol(
  self : RankGroupOwner,
  rule : RankGroupRule,
  rank : Int,
) -> T raise RankGroupError {
  match self.lifecycle {
    Faulted => raise GroupFaulted(self.failure_reason)
    Closed => raise InvalidLifecycle(expected=Executing, actual=Closed)
    Completed => raise InvalidLifecycle(expected=Executing, actual=Completed)
    Ready | Executing => {
      self.protocol_fault(rule, rank)
      raise ProtocolFault(rule)
    }
  }
}

///|
fn RankGroupOwner::find_plan_buffer(
  self : RankGroupOwner,
  identity : @worker_protocol.PlanBufferIdentity,
) -> Int {
  for index in 0..<self.plan_buffer_identities.length() {
    if self.plan_buffer_identities[index].same_owner(identity) {
      return index
    }
  }
  -1
}

///|
/// Rederives the current allocation-free submission capability only after
/// authenticating the exact scheduler-owned plan and live owner epoch.
pub fn RankGroupOwner::reauthenticate_submitted(
  self : RankGroupOwner,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> SubmittedRankGroupPlan raise RankGroupError {
  self.require_lifecycle(Executing)
  let submitted = SubmittedRankGroupPlan(self, self.epoch)
  ignore(submitted.authenticate_plan(plan, Executing))
  submitted
}

///|
pub fn RankGroupOwner::begin(
  self : RankGroupOwner,
  plan : @worker_protocol.SubmittedSchedulePlan,
  started_tick~ : UInt64,
  deadline_tick~ : UInt64,
) -> SubmittedRankGroupPlan raise RankGroupError {
  self.require_lifecycle(Ready)
  self.start_epoch()
  let plan_buffer = plan.buffer_identity() catch {
    _ => return self.reject_protocol(Submission, -1)
  }
  let sequence = plan.sequence_wire() catch {
    _ => return self.reject_protocol(Submission, -1)
  }
  let model_generation = plan.model_plan_generation_wire() catch {
    _ => return self.reject_protocol(Submission, -1)
  }
  self.active_sequence_value = sequence
  let plan_buffer_index = self.find_plan_buffer(plan_buffer)
  if plan_buffer_index < 0 {
    return self.reject_protocol(PlanBufferSet, -1)
  }
  self.active_plan_buffer_index = plan_buffer_index
  if model_generation != self.model_plan_generation_value {
    return self.reject_protocol(ModelPlanGeneration, -1)
  }
  if sequence != self.previous_sequence_value + 1UL {
    return self.reject_protocol(PlanSequence, -1)
  }
  if deadline_tick == 0UL || deadline_tick <= started_tick {
    return self.reject_protocol(Deadline, -1)
  }
  self.active_started_tick = started_tick
  self.active_deadline_tick = deadline_tick
  self.lifecycle = Executing
  SubmittedRankGroupPlan(self, self.epoch)
}

///|
fn SubmittedRankGroupPlan::authenticate(
  self : SubmittedRankGroupPlan,
  expected : RankGroupLifecycle,
) -> RankGroupOwner raise RankGroupError {
  let owner = self.0
  if self.1 != owner.epoch {
    if owner.lifecycle == Executing {
      return owner.reject_protocol(Submission, -1)
    }
    raise Invalid(Submission)
  }
  owner.require_lifecycle(expected)
  owner
}

///|
fn SubmittedRankGroupPlan::authenticate_plan(
  self : SubmittedRankGroupPlan,
  plan : @worker_protocol.SubmittedSchedulePlan,
  expected : RankGroupLifecycle,
) -> RankGroupOwner raise RankGroupError {
  let owner = self.authenticate(expected)
  let identity = plan.buffer_identity() catch {
    _ => return owner.reject_protocol(Submission, -1)
  }
  let index = owner.active_plan_buffer_index
  if index < 0 ||
    index >= owner.plan_buffer_identities.length() ||
    !owner.plan_buffer_identities[index].same_owner(identity) {
    return owner.reject_protocol(Submission, -1)
  }
  let sequence = plan.sequence_wire() catch {
    _ => return owner.reject_protocol(Submission, -1)
  }
  let model_generation = plan.model_plan_generation_wire() catch {
    _ => return owner.reject_protocol(Submission, -1)
  }
  if sequence != owner.active_sequence_value {
    return owner.reject_protocol(PlanSequence, -1)
  }
  if model_generation != owner.model_plan_generation_value {
    return owner.reject_protocol(ModelPlanGeneration, -1)
  }
  owner
}

///|
pub fn SubmittedRankGroupPlan::validate_plan(
  self : SubmittedRankGroupPlan,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> Unit raise RankGroupError {
  ignore(self.authenticate_plan(plan, Executing))
}

///|
pub fn SubmittedRankGroupPlan::sequence_value(
  self : SubmittedRankGroupPlan,
) -> UInt64 raise RankGroupError {
  self.authenticate(Executing).active_sequence_value
}

///|
pub fn SubmittedRankGroupPlan::deadline_tick(
  self : SubmittedRankGroupPlan,
) -> UInt64 raise RankGroupError {
  self.authenticate(Executing).active_deadline_tick
}

///|
pub fn SubmittedRankGroupPlan::lease(
  self : SubmittedRankGroupPlan,
  plan : @worker_protocol.SubmittedSchedulePlan,
  rank : Int,
) -> RankPlanLease raise RankGroupError {
  let owner = self.authenticate_plan(plan, Executing)
  if rank < 0 || rank >= owner.world_size {
    return owner.reject_protocol(Rank, rank)
  }
  RankPlanLease(owner, self.1, rank)
}
