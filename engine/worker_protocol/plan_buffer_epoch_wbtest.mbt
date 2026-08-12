///|
test "epoch overflow leaves the authenticated buffer retired" {
  let limits = WorkerProtocolLimits::new(
    max_prefill_rows=1,
    max_decode_rows=1,
    max_plan_rows=2,
    max_plan_tokens=2,
    max_plan_pages=2,
    max_capabilities=2,
    max_completion_slots=2,
    max_sequence_tokens=8,
    max_token_id=64,
  )
  let buffer = SchedulePlanBuffer::new(limits)
  buffer.epoch = 0xffffffffffffffffUL
  let submitted = SubmittedSchedulePlan(buffer, buffer.epoch)
  buffer.sequence_value = 1UL
  buffer.model_plan_generation_value = 1UL
  buffer.token_budget = 0
  buffer.state = Submitted
  let retired = buffer.retire(submitted)
  try buffer.reset(retired) catch {
    IdentityOverflow(PlanBufferEpochIdentity) => ()
    _ => fail("expected epoch overflow")
  } noraise {
    _ => fail("expected reset to fail")
  }
  assert_eq(buffer.state(), Retired)
  assert_eq(buffer.epoch(), 0xffffffffffffffffUL)
  assert_eq(buffer.token_count, 0)
  assert_eq(buffer.page_count, 0)
}
