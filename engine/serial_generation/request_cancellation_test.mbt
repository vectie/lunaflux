///|
test "request cancellation is an explicit immutable submission plan not a polling ceiling" {
  let plan = @serial_generation.SerialRequestCancellation::new(
    request_index=1,
    after_submitted_steps=3,
  )
  plan.validate_request_count(2)
  assert_eq(plan.after_steps_for(0), 0)
  assert_eq(plan.after_steps_for(1), 3)
  assert_eq(plan.after_steps_for(2), 0)
  for (index, steps) in [(-1, 1), (0, 0), (0, -1)] {
    try
      @serial_generation.SerialRequestCancellation::new(
        request_index=index,
        after_submitted_steps=steps,
      )
    catch {
      @serial_generation.SerialGenerationError::InvalidLimit => ()
      error => raise error
    } noraise {
      _ => fail("invalid cancellation plan accepted")
    }
  }
  try plan.validate_request_count(1) catch {
    @serial_generation.SerialGenerationError::InvalidLimit => ()
    error => raise error
  } noraise {
    _ => fail("cancelled a request outside the prepared queue")
  }
}
