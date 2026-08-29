///|
test "current_submitted returns only the exact submitted epoch" {
  let owner = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  try owner.current_submitted() catch {
    InvalidPlanBufferState(expected=Submitted, actual=Writable) => ()
    _ => fail("expected writable owner rejection")
  } noraise {
    _ => fail("writable owner returned a submitted handle")
  }
  let submitted = submitted_plan_for_completion(owner)
  let current = owner.current_submitted()
  assert_eq(current.sequence_wire(), submitted.sequence_wire())
  assert_eq(current.model_plan_generation_wire(), 1UL)
  let retired = owner.retire(submitted)
  try owner.current_submitted() catch {
    InvalidPlanBufferState(expected=Submitted, actual=Retired) => ()
    _ => fail("expected retired owner rejection")
  } noraise {
    _ => fail("retired owner returned a submitted handle")
  }
  owner.reset(retired)
}

///|
test "submitted plan slot retains one exact owner epoch without copying" {
  let owner = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let slot = @worker_protocol.SubmittedSchedulePlanSlot::new()
  try slot.current() catch {
    StalePlanBufferEpoch => ()
    _ => fail("empty submitted-plan slot reported the wrong error")
  } noraise {
    _ => fail("empty submitted-plan slot returned a plan")
  }
  let submitted = submitted_plan_for_completion(owner)
  slot.retain(submitted)
  assert_eq(slot.current().sequence_wire(), submitted.sequence_wire())
  let retired = owner.retire(submitted)
  try slot.current() catch {
    InvalidPlanBufferState(expected=Submitted, actual=Retired) => ()
    _ => fail("retained plan epoch reported the wrong stale error")
  } noraise {
    _ => fail("retained plan survived owner retirement")
  }
  slot.clear()
  owner.reset(retired)
}
