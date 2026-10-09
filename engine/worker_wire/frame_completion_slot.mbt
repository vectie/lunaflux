///|
/// Startup-allocated retention of an exact plan/writer lease and its published
/// completion. Value-type wire views are reconstructed without heap-boxed
/// Option values. This slot does not own or extend the underlying frame epoch.
pub struct FrameCompletionSlot {
  priv mut plan_owner : PlanFrameBuffer?
  priv mut completion_owner : CompletionFrameBuffer?
  priv mut plan_epoch : UInt64
  priv mut completion_epoch : UInt64
  priv mut expected_entries : Int
  priv mut active : Bool
  priv mut published : Bool
}

///|
pub fn FrameCompletionSlot::new() -> Self {
  {
    plan_owner: None,
    completion_owner: None,
    plan_epoch: 0UL,
    completion_epoch: 0UL,
    expected_entries: 0,
    active: false,
    published: false,
  }
}

///|
/// Retain the exact writer, never reopen or substitute its epoch.
pub fn FrameCompletionSlot::bind(
  self : Self,
  plan : ValidatedPlanFrame,
  writer : CompletionFrameWriter,
) -> Unit raise WorkerWireError {
  if self.active {
    raise CompletionFrameWriterAlreadyOpen
  }
  writer.validate_plan(plan)
  self.plan_owner = Some(plan.0)
  self.completion_owner = Some(writer.0)
  self.plan_epoch = plan.1
  self.completion_epoch = writer.1
  self.expected_entries = writer.4
  self.published = false
  self.active = true
}

///|
pub fn FrameCompletionSlot::plan(
  self : Self,
) -> ValidatedPlanFrame raise WorkerWireError {
  guard self.active && self.plan_owner is Some(owner) else { raise StaleFrame }
  let plan = ValidatedPlanFrame(owner, self.plan_epoch)
  ignore(plan.owner())
  plan
}

///|
pub fn FrameCompletionSlot::writer(
  self : Self,
) -> CompletionFrameWriter raise WorkerWireError {
  guard self.active &&
    self.plan_owner is Some(plan_owner) &&
    self.completion_owner is Some(completion_owner) else {
    raise CompletionFrameWriterNotOpen
  }
  let writer = CompletionFrameWriter(
    completion_owner,
    self.completion_epoch,
    plan_owner,
    self.plan_epoch,
    self.expected_entries,
  )
  ignore(writer.owner())
  writer
}

///|
pub fn FrameCompletionSlot::submit(
  self : Self,
) -> ValidatedCompletionFrame raise WorkerWireError {
  let completion = self.writer().submit()
  self.plan_owner = None
  self.active = false
  self.published = true
  completion
}

///|
/// A by-value completion accessor; callers first poll a Boolean readiness
/// result instead of constructing an Option[ValidatedCompletionFrame].
pub fn FrameCompletionSlot::completion(
  self : Self,
) -> ValidatedCompletionFrame raise WorkerWireError {
  guard self.published && self.completion_owner is Some(owner) else {
    raise StaleFrame
  }
  let completion = ValidatedCompletionFrame(owner, self.completion_epoch)
  ignore(completion.owner())
  completion
}

///|
/// Abort an active lease and drop retained references. Repeated calls are safe.
pub fn FrameCompletionSlot::close(self : Self) -> Unit raise WorkerWireError {
  if self.active {
    self.writer().abort()
  }
  self.active = false
  self.published = false
  self.plan_owner = None
  self.completion_owner = None
}
