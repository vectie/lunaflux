///|
/// Starts an immutable cursor over the exact stage sequence carried by the
/// generic plan.
pub fn JointDiffusionStageMachine::start(
  plan : @joint_diffusion_plan.JointDiffusionPlan,
) -> JointDiffusionStageMachine raise JointDiffusionReferenceError {
  let stages = plan.stages()
  if plan.schema_version() != 1 || stages.is_empty() || stages.length() > 32 {
    raise InvalidPlanContract
  }
  { plan_digest: plan.plan_digest(), stages, next_index: 0, cancelled: false, }
}

///|
fn JointDiffusionStageMachine::ensure_active(
  self : Self,
) -> Unit raise JointDiffusionReferenceError {
  if self.cancelled {
    raise StageMachineCancelled
  }
  if self.next_index >= self.stages.length() {
    raise StageMachineComplete
  }
}

///|
/// Completes exactly the current stage and returns the next immutable state.
pub fn JointDiffusionStageMachine::advance(
  self : Self,
  actual : @joint_diffusion_plan.JointDiffusionStageKind,
) -> JointDiffusionStageMachine raise JointDiffusionReferenceError {
  self.ensure_active()
  let expected = self.stages[self.next_index].kind()
  if actual != expected {
    raise StageMismatch(expected~, actual~)
  }
  { ..self, next_index: self.next_index + 1, }
}

///|
/// Skips only a stage explicitly marked optional by the plan.
pub fn JointDiffusionStageMachine::skip_optional(
  self : Self,
) -> JointDiffusionStageMachine raise JointDiffusionReferenceError {
  self.ensure_active()
  let stage = self.stages[self.next_index]
  if stage.presence() != Optional {
    raise RequiredStageCannotBeSkipped(stage.kind())
  }
  { ..self, next_index: self.next_index + 1, }
}

///|
/// Cancels an incomplete machine. Cancellation is idempotent and terminal.
pub fn JointDiffusionStageMachine::cancel(
  self : Self,
) -> JointDiffusionStageMachine raise JointDiffusionReferenceError {
  if self.next_index >= self.stages.length() {
    raise CompletedMachineCannotBeCancelled
  }
  if self.cancelled {
    self
  } else {
    { ..self, cancelled: true, }
  }
}

///|
pub fn JointDiffusionStageMachine::plan_digest(self : Self) -> @spec.PlanDigest {
  self.plan_digest
}

///|
pub fn JointDiffusionStageMachine::completed_stage_count(self : Self) -> Int {
  self.next_index
}

///|
pub fn JointDiffusionStageMachine::is_complete(self : Self) -> Bool {
  !self.cancelled && self.next_index == self.stages.length()
}

///|
pub fn JointDiffusionStageMachine::is_cancelled(self : Self) -> Bool {
  self.cancelled
}

///|
pub fn JointDiffusionStageMachine::next_stage(
  self : Self,
) -> @joint_diffusion_plan.JointDiffusionStage? {
  if self.cancelled {
    None
  } else {
    self.stages.get(self.next_index)
  }
}
