///|
test "output workspace reuses selected hidden after the first head" {
  let plan = output_layout(39744, 5376, 3, 37296, 414, 2000000000L)
  assert_eq(plan.parameters, 39744L * 5376L * 2L)
  assert_eq(plan.selected, plan.parameters + 3L * 5376L * 4L)
  assert_eq(plan.bytes, plan.selected + 37296L * 5376L * 4L)
  ignore(output_layout(39744, 5376, 3, 37296, 414, plan.bytes))
  try output_layout(39744, 5376, 3, 37296, 414, plan.bytes - 1L) catch {
    @device.DeviceError::InvalidArgument(AllocationCreate) => ()
    _ => fail("unexpected budget error")
  } noraise {
    _ => fail("short budget accepted")
  }
}
