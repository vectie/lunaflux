///|
fn validate_world_size(world_size : Int) -> Unit raise RankGroupError {
  if world_size < 2 || world_size > MAX_RANK_GROUP_WORLD_SIZE {
    raise Invalid(WorldSize)
  }
}

///|
fn validate_rank(rank : Int, world_size : Int) -> Unit raise RankGroupError {
  if rank < 0 || rank >= world_size {
    raise Invalid(Rank)
  }
}

///|
fn copy_plan_buffer_identities(
  source : ArrayView[@worker_protocol.PlanBufferIdentity],
) -> FixedArray[@worker_protocol.PlanBufferIdentity] raise RankGroupError {
  if source.is_empty() || source.length() > 2 {
    raise Invalid(PlanBufferSet)
  }
  for left in 0..<source.length() {
    for right in 0..<left {
      if source[left].same_owner(source[right]) {
        raise Invalid(PlanBufferSet)
      }
    }
  }
  let copied = FixedArray::make(source.length(), source[0])
  for index, identity in source {
    copied[index] = identity
  }
  copied
}

///|
pub fn RankGroupOwner::new(
  identity~ : @spec.ModelIdentity,
  model_plan_generation~ : @worker_protocol.ModelPlanGeneration,
  generation~ : RankGroupGeneration,
  world_size~ : Int,
  previous_plan_sequence_value~ : UInt64,
  plan_buffer_identities~ : ArrayView[@worker_protocol.PlanBufferIdentity],
) -> RankGroupOwner raise RankGroupError {
  validate_world_size(world_size)
  if previous_plan_sequence_value == 0xffffffffffffffffUL {
    raise Invalid(PlanSequence)
  }
  let admitted_plan_buffers = copy_plan_buffer_identities(
    plan_buffer_identities,
  )
  {
    identity,
    model_plan_generation_value: model_plan_generation.value(),
    generation,
    world_size,
    plan_buffer_identities: admitted_plan_buffers,
    lifecycle: Ready,
    epoch: 0UL,
    previous_sequence_value: previous_plan_sequence_value,
    active_sequence_value: 0UL,
    active_started_tick: 0UL,
    active_deadline_tick: 0UL,
    active_plan_buffer_index: -1,
    follower_acks: FixedArray::make(world_size, false),
    drained_ranks: FixedArray::make(world_size, false),
    lost_ranks: FixedArray::make(world_size, false),
    follower_ack_count: 0,
    failure_reason: ProtocolViolation(Submission),
    failure_rank: -1,
    failure_published: false,
    drain_count: 0,
  }
}

///|
pub fn RankGroupOwner::lifecycle(self : RankGroupOwner) -> RankGroupLifecycle {
  self.lifecycle
}

///|
pub fn RankGroupOwner::identity(self : RankGroupOwner) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn RankGroupOwner::generation(self : RankGroupOwner) -> RankGroupGeneration {
  self.generation
}

///|
pub fn RankGroupOwner::world_size(self : RankGroupOwner) -> Int {
  self.world_size
}

///|
pub fn RankGroupOwner::previous_plan_sequence_value(
  self : RankGroupOwner,
) -> UInt64 {
  self.previous_sequence_value
}

///|
fn RankGroupOwner::require_lifecycle(
  self : RankGroupOwner,
  expected : RankGroupLifecycle,
) -> Unit raise RankGroupError {
  if self.lifecycle != expected {
    raise InvalidLifecycle(expected~, actual=self.lifecycle)
  }
}

///|
fn RankGroupOwner::clear_plan_state(self : RankGroupOwner) -> Unit {
  self.active_sequence_value = 0UL
  self.active_started_tick = 0UL
  self.active_deadline_tick = 0UL
  self.active_plan_buffer_index = -1
  self.follower_ack_count = 0
  for rank in 0..<self.world_size {
    self.follower_acks[rank] = false
    self.drained_ranks[rank] = false
    self.lost_ranks[rank] = false
  }
  self.failure_rank = -1
  self.failure_published = false
  self.drain_count = 0
}

///|
fn RankGroupOwner::start_epoch(
  self : RankGroupOwner,
) -> Unit raise RankGroupError {
  if self.epoch == 0xffffffffffffffffUL {
    raise Invalid(GroupEpoch)
  }
  self.epoch += 1UL
  self.clear_plan_state()
}

///|
pub fn RankGroupOwner::close(
  self : RankGroupOwner,
) -> Unit raise RankGroupError {
  self.require_lifecycle(Ready)
  self.lifecycle = Closed
}
