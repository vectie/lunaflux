///|
test "validated profile views are readable only through checked public APIs" {
  let static_plan = profile_static_plan()
  let memory_plan = profile_memory_plan(static_plan~)
  let shape = @device_profile.ExactExecutionShape::full_prefill(
    batch_rows=1,
    sequence_tokens=2,
  )
  let kernel_shape = shape.kernel_shape()
  assert_eq(kernel_shape.batch_rows(), 1)
  assert_eq(kernel_shape.sequence_tokens(), 2)
  assert_eq(kernel_shape.token_rows(), 2L)
  let profile = @device_profile.build(static_plan, memory_plan, shape)
  let token_staging = profile.token_staging()
  assert_eq(token_staging.id().as_int(), 0)
  assert_eq(token_staging.offset_bytes(), 0L)
  let operation = profile.operation(@plan.OperationId::new(0)).unwrap()
  assert_eq(operation.operation_id().as_int(), 0)
  assert_eq(operation.kernel_shape(), kernel_shape)
  let output = operation.outputs()[0]
  assert_eq(output.reference().as_int(), 0)
  assert_eq(output.token_rows(), 2L)
  assert_eq(profile.region(output.region().id()), Some(output.region()))
  let final_output = profile.final_output()
  assert_eq(final_output.selected_rows().length(), 1)
  let row = final_output.selected_rows()[0]
  assert_eq(row.batch_index(), 0)
  assert_eq(row.backing_region(), final_output.value().region().id())
  assert_eq(row.end_offset_bytes(), row.offset_bytes() + row.byte_length())
}

///|
test "checked shape and region-id constructors reject hostile values" {
  try
    @device_profile.ExactExecutionShape::full_prefill(
      batch_rows=0,
      sequence_tokens=1,
    )
  catch {
    InvalidShape(NonPositiveBatchRows) => ()
    _ => fail("unexpected invalid-shape error")
  } noraise {
    _ => fail("expected a non-positive shape rejection")
  }
  try @device_profile.ProfileRegionId::from_stable_int(-1) catch {
    InvalidRegion(id, IdentifierOverflow) => assert_eq(id.as_int(), 0)
    _ => fail("unexpected invalid-region-id error")
  } noraise {
    _ => fail("expected a negative region-id rejection")
  }
  assert_eq(
    @device_profile.ProfileRegionId::from_stable_int(2147483647).as_int(),
    2147483647,
  )
}
