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
