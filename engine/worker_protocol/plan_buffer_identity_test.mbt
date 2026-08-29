///|
test "plan-buffer identity is stable across epochs and distinct across owners" {
  let first = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  let second = @worker_protocol.SchedulePlanBuffer::new(protocol_limits())
  assert_true(first.identity().same_owner(first.identity()))
  assert_false(first.identity().same_owner(second.identity()))
  let submitted = submitted_plan_for_completion(first)
  assert_true(submitted.buffer_identity().same_owner(first.identity()))
  let retired = first.retire(submitted)
  first.reset(retired)
  assert_true(first.identity().same_owner(first.identity()))
}
