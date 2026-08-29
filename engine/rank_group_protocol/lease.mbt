///|
fn RankPlanLease::authenticate(
  self : RankPlanLease,
) -> RankGroupOwner raise RankGroupError {
  let owner = self.0
  if self.1 != owner.epoch {
    if owner.lifecycle == Executing {
      return owner.reject_protocol(Submission, self.2)
    }
    raise Invalid(Submission)
  }
  owner.require_lifecycle(Executing)
  if self.2 < 0 || self.2 >= owner.world_size {
    return owner.reject_protocol(Rank, self.2)
  }
  owner
}

///|
pub fn RankPlanLease::rank(self : RankPlanLease) -> Int raise RankGroupError {
  ignore(self.authenticate())
  self.2
}

///|
pub fn RankPlanLease::world_size(
  self : RankPlanLease,
) -> Int raise RankGroupError {
  self.authenticate().world_size
}

///|
pub fn RankPlanLease::group_generation_value(
  self : RankPlanLease,
) -> UInt64 raise RankGroupError {
  self.authenticate().generation.value()
}

///|
pub fn RankPlanLease::model_identity(
  self : RankPlanLease,
) -> @spec.ModelIdentity raise RankGroupError {
  self.authenticate().identity
}

///|
pub fn RankPlanLease::model_plan_generation_value(
  self : RankPlanLease,
) -> UInt64 raise RankGroupError {
  self.authenticate().model_plan_generation_value
}

///|
pub fn RankPlanLease::plan_sequence_value(
  self : RankPlanLease,
) -> UInt64 raise RankGroupError {
  self.authenticate().active_sequence_value
}

///|
pub fn RankPlanLease::deadline_tick(
  self : RankPlanLease,
) -> UInt64 raise RankGroupError {
  self.authenticate().active_deadline_tick
}

///|
pub fn RankPlanLease::follower_ack(
  self : RankPlanLease,
) -> RankExecutionAck raise RankGroupError {
  let owner = self.authenticate()
  if self.2 == 0 {
    return owner.reject_protocol(FollowerRank, self.2)
  }
  RankExecutionAck(
    owner.generation.value(),
    owner.active_sequence_value,
    owner.model_plan_generation_value,
    self.2,
    owner.world_size,
  )
}

///|
pub fn RankExecutionAck::from_wire(
  group_generation_value~ : UInt64,
  plan_sequence_value~ : UInt64,
  model_plan_generation_value~ : UInt64,
  rank~ : Int,
  world_size~ : Int,
) -> RankExecutionAck raise RankGroupError {
  validate_world_size(world_size)
  if group_generation_value == 0UL {
    raise Invalid(GroupGeneration)
  }
  if plan_sequence_value == 0UL {
    raise Invalid(PlanSequence)
  }
  if model_plan_generation_value == 0UL {
    raise Invalid(ModelPlanGeneration)
  }
  validate_rank(rank, world_size)
  if rank == 0 {
    raise Invalid(FollowerRank)
  }
  RankExecutionAck(
    group_generation_value, plan_sequence_value, model_plan_generation_value, rank,
    world_size,
  )
}

///|
pub fn RankExecutionAck::rank(self : RankExecutionAck) -> Int {
  self.3
}
