///|
pub fn SubmittedSchedulePlanSlot::new() -> SubmittedSchedulePlanSlot {
  { owner: None, epoch: 0UL }
}

///|
/// Retains the exact currently submitted owner and epoch in startup storage.
pub fn SubmittedSchedulePlanSlot::retain(
  self : SubmittedSchedulePlanSlot,
  plan : SubmittedSchedulePlan,
) -> Unit raise WorkerProtocolError {
  ignore(plan.authenticate())
  self.owner = Some(plan.0)
  self.epoch = plan.1
}

///|
pub fn SubmittedSchedulePlanSlot::clear(
  self : SubmittedSchedulePlanSlot,
) -> Unit {
  self.owner = None
  self.epoch = 0UL
}

///|
/// Reconstructs the stack-valued immutable handle and reauthenticates it.
pub fn SubmittedSchedulePlanSlot::current(
  self : SubmittedSchedulePlanSlot,
) -> SubmittedSchedulePlan raise WorkerProtocolError {
  guard self.owner is Some(owner) && self.epoch != 0UL else {
    raise StalePlanBufferEpoch
  }
  let plan = SubmittedSchedulePlan(owner, self.epoch)
  ignore(plan.authenticate())
  plan
}
