///|
/// Immutable startup binding for the scheduler's two reusable plan-buffer
/// owners. The binding contains no plan payload, epoch, sequence, or mutation
/// authority and has no public constructor.
pub struct SchedulerPlanBufferBinding {
  priv plan_a_identity : @worker_protocol.PlanBufferIdentity
  priv plan_b_identity : @worker_protocol.PlanBufferIdentity
}

///|
fn scheduler_plan_buffer_binding(
  plan_a_identity : @worker_protocol.PlanBufferIdentity,
  plan_b_identity : @worker_protocol.PlanBufferIdentity,
) -> SchedulerPlanBufferBinding {
  if plan_a_identity.same_owner(plan_b_identity) {
    abort("scheduler plan-buffer binding identity invariant")
  }
  { plan_a_identity, plan_b_identity }
}

///|
/// Returns the exact number of ordered identities retained by this binding.
/// Production schedulers own one A/B pair for bounded plan overlap.
pub fn SchedulerPlanBufferBinding::count(
  self : SchedulerPlanBufferBinding,
) -> Int {
  ignore(self)
  2
}

///|
/// Returns the stable identity of the scheduler's A plan-buffer owner.
pub fn SchedulerPlanBufferBinding::plan_a_identity(
  self : SchedulerPlanBufferBinding,
) -> @worker_protocol.PlanBufferIdentity {
  self.plan_a_identity
}

///|
/// Returns the stable identity of the scheduler's B plan-buffer owner.
pub fn SchedulerPlanBufferBinding::plan_b_identity(
  self : SchedulerPlanBufferBinding,
) -> @worker_protocol.PlanBufferIdentity {
  self.plan_b_identity
}

///|
/// Returns the scheduler's immutable startup binding. Repeated calls return
/// the same binding and perform no plan-epoch or sequence-dependent work.
pub fn Scheduler::plan_buffer_binding(
  self : Scheduler,
) -> SchedulerPlanBufferBinding {
  self.plan_buffer_binding
}
