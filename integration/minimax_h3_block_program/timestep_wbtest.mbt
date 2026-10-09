///|
test "timestep workspace separates frequency hidden and embedding" {
  let plan = timestep_workspace(3, 256, 5376, 2688, 100000L)
  assert_eq(plan.hidden, 3072L)
  assert_eq(plan.output, 67584L)
  assert_eq(plan.bytes, 99840L)
  try timestep_workspace(3, 256, 5376, 2688, 99839L) catch {
    @device.DeviceError::InvalidArgument(AllocationCreate) => ()
    _ => fail("wrong budget error")
  } noraise {
    _ => fail("short budget accepted")
  }
}
