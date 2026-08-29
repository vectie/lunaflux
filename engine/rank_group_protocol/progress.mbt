///|
pub fn RankGroupOwner::accept_follower_ack(
  self : RankGroupOwner,
  submitted : SubmittedRankGroupPlan,
  ack : RankExecutionAck,
) -> Unit raise RankGroupError {
  let owner = submitted.authenticate(Executing) catch {
    _ => return self.reject_protocol(Submission, ack.3)
  }
  if !physical_equal(owner, self) {
    return self.reject_protocol(Submission, ack.3)
  }
  if ack.0 != self.generation.value() {
    return self.reject_protocol(GroupGeneration, ack.3)
  }
  if ack.1 != self.active_sequence_value {
    return self.reject_protocol(PlanSequence, ack.3)
  }
  if ack.2 != self.model_plan_generation_value {
    return self.reject_protocol(ModelPlanGeneration, ack.3)
  }
  if ack.4 != self.world_size {
    return self.reject_protocol(WorldSize, ack.3)
  }
  if ack.3 <= 0 || ack.3 >= self.world_size {
    return self.reject_protocol(FollowerRank, ack.3)
  }
  if self.follower_acks[ack.3] {
    return self.reject_protocol(DuplicateFollowerAck, ack.3)
  }
  self.follower_acks[ack.3] = true
  self.follower_ack_count += 1
}

///|
pub fn RankGroupOwner::followers_complete(
  self : RankGroupOwner,
  submitted : SubmittedRankGroupPlan,
) -> Bool raise RankGroupError {
  let owner = submitted.authenticate(Executing)
  if !physical_equal(owner, self) {
    return self.reject_protocol(Submission, -1)
  }
  self.follower_ack_count == self.world_size - 1
}

///|
pub fn RankGroupOwner::complete_leader(
  self : RankGroupOwner,
  leader : RankPlanLease,
  plan : @worker_protocol.SubmittedSchedulePlan,
  completion : @worker_protocol.SubmittedCompletion,
) -> CompletedRankGroupPlan raise RankGroupError {
  let owner = leader.authenticate() catch {
    _ => return self.reject_protocol(Submission, -1)
  }
  if !physical_equal(owner, self) {
    return self.reject_protocol(Submission, leader.2)
  }
  if leader.2 != 0 {
    return self.reject_protocol(LeaderRank, leader.2)
  }
  let submitted = SubmittedRankGroupPlan(self, leader.1)
  ignore(submitted.authenticate_plan(plan, Executing))
  if self.follower_ack_count != self.world_size - 1 {
    raise NotReady(FollowersPending)
  }
  completion.validate_for(plan) catch {
    _ => return self.reject_protocol(Completion, 0)
  }
  self.lifecycle = Completed
  CompletedRankGroupPlan(self, leader.1)
}

///|
fn CompletedRankGroupPlan::authenticate(
  self : CompletedRankGroupPlan,
) -> RankGroupOwner raise RankGroupError {
  let owner = self.0
  if self.1 != owner.epoch || owner.lifecycle != Completed {
    raise Invalid(Submission)
  }
  owner
}

///|
/// Rederives the post-validation retirement capability from copied epoch
/// evidence without re-reading scheduler-owned plan or completion values.
pub fn RankGroupOwner::reauthenticate_completed(
  self : RankGroupOwner,
  epoch : UInt64,
) -> CompletedRankGroupPlan raise RankGroupError {
  self.require_lifecycle(Completed)
  let completed = CompletedRankGroupPlan(self, epoch)
  ignore(completed.authenticate())
  completed
}

///|
pub fn CompletedRankGroupPlan::validate_completion(
  self : CompletedRankGroupPlan,
  plan : @worker_protocol.SubmittedSchedulePlan,
  completion : @worker_protocol.SubmittedCompletion,
) -> Unit raise RankGroupError {
  let owner = self.authenticate()
  let submitted = SubmittedRankGroupPlan(owner, self.1)
  ignore(submitted.authenticate_plan(plan, Completed))
  completion.validate_for(plan) catch {
    _ => raise Invalid(Completion)
  }
}

///|
pub fn RankGroupOwner::retire_completed(
  self : RankGroupOwner,
  completed : CompletedRankGroupPlan,
) -> Unit raise RankGroupError {
  let owner = completed.authenticate()
  if !physical_equal(owner, self) {
    raise Invalid(Submission)
  }
  self.previous_sequence_value = self.active_sequence_value
  self.clear_plan_state()
  self.lifecycle = Ready
}

///|
pub fn RankGroupOwner::expire_if_due(
  self : RankGroupOwner,
  submitted : SubmittedRankGroupPlan,
  now_tick : UInt64,
) -> Bool raise RankGroupError {
  let owner = submitted.authenticate(Executing) catch {
    _ => return self.reject_protocol(Submission, -1)
  }
  if !physical_equal(owner, self) {
    return self.reject_protocol(Submission, -1)
  }
  if now_tick < self.active_started_tick {
    return self.reject_protocol(TimeRegression, -1)
  }
  if now_tick < self.active_deadline_tick {
    return false
  }
  self.fault(DeadlineExpired, -1, false)
  true
}

///|
pub fn RankGroupOwner::report_rank_loss(
  self : RankGroupOwner,
  submitted : SubmittedRankGroupPlan,
  rank : Int,
) -> Unit raise RankGroupError {
  let owner = submitted.authenticate(Executing) catch {
    _ => return self.reject_protocol(Submission, rank)
  }
  if !physical_equal(owner, self) {
    return self.reject_protocol(Submission, rank)
  }
  if rank < 0 || rank >= self.world_size {
    return self.reject_protocol(Rank, rank)
  }
  self.fault(RankLost, rank, true)
}

///|
pub fn RankGroupOwner::report_collective_failure(
  self : RankGroupOwner,
  submitted : SubmittedRankGroupPlan,
  rank : Int,
) -> Unit raise RankGroupError {
  let owner = submitted.authenticate(Executing) catch {
    _ => return self.reject_protocol(Submission, rank)
  }
  if !physical_equal(owner, self) {
    return self.reject_protocol(Submission, rank)
  }
  if rank < 0 || rank >= self.world_size {
    return self.reject_protocol(Rank, rank)
  }
  self.fault(CollectiveFailure, rank, false)
}

///|
pub fn RankGroupOwner::report_rank_execution_failure(
  self : RankGroupOwner,
  submitted : SubmittedRankGroupPlan,
  rank : Int,
) -> Unit raise RankGroupError {
  let owner = submitted.authenticate(Executing) catch {
    _ => return self.reject_protocol(Submission, rank)
  }
  if !physical_equal(owner, self) {
    return self.reject_protocol(Submission, rank)
  }
  if rank < 0 || rank >= self.world_size {
    return self.reject_protocol(Rank, rank)
  }
  self.fault(RankExecutionFailure, rank, false)
}

///|
/// Authenticates the exact live scheduler plan before a parent supervisor
/// converts a local protocol/codec failure into terminal rank execution loss.
pub fn RankGroupOwner::report_supervisor_failure(
  self : RankGroupOwner,
  plan : @worker_protocol.SubmittedSchedulePlan,
  rank : Int,
) -> Unit raise RankGroupError {
  let submitted = self.reauthenticate_submitted(plan)
  self.report_rank_execution_failure(submitted, rank)
}

///|
pub fn RankGroupOwner::report_supervisor_rank_loss(
  self : RankGroupOwner,
  plan : @worker_protocol.SubmittedSchedulePlan,
  rank : Int,
) -> Unit raise RankGroupError {
  let submitted = self.reauthenticate_submitted(plan)
  self.report_rank_loss(submitted, rank)
}
