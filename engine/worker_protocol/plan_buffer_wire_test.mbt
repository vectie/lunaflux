///|
test "scalar submit preserves authenticated identity and sequencing" {
  let limits = protocol_limits()
  let buffer = try! @worker_protocol.SchedulePlanBuffer::new(limits)
  append_mixed_buffer(buffer, allocator())
  let submitted = buffer.submit_wire(
    sequence_value=1UL,
    model_plan_generation_value=1UL,
    token_budget=3,
    has_previous=false,
    previous_sequence_value=0UL,
    loaded_generation_value=1UL,
  )
  assert_eq(try! submitted.sequence_wire(), 1UL)
  assert_eq(try! submitted.model_plan_generation_wire(), 1UL)
  assert_eq(try! submitted.sequence().value(), 1UL)
  assert_eq(try! submitted.model_plan_generation().value(), 1UL)
}

///|
test "scalar submit rejects invalid and stale wire identities before freezing" {
  let limits = protocol_limits()
  let buffer = try! @worker_protocol.SchedulePlanBuffer::new(limits)
  append_mixed_buffer(buffer, allocator())
  try
    buffer.submit_wire(
      sequence_value=0UL,
      model_plan_generation_value=1UL,
      token_budget=3,
      has_previous=false,
      previous_sequence_value=0UL,
      loaded_generation_value=1UL,
    )
  catch {
    error => assert_eq(error, InvalidIdentity(PlanSequenceIdentity))
  } noraise {
    _ => fail("expected zero sequence rejection")
  }
  assert_eq(buffer.state(), Writable)
  let submitted = buffer.submit_wire(
    sequence_value=2UL,
    model_plan_generation_value=1UL,
    token_budget=3,
    has_previous=true,
    previous_sequence_value=1UL,
    loaded_generation_value=1UL,
  )
  assert_eq(try! submitted.sequence_wire(), 2UL)
}
