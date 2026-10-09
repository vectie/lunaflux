///|
test "H3 timestep staging has bounded reusable F32 storage" {
  let plan = TimestepMlpLayout::new(
    rows=3,
    input_width=256,
    hidden_width=5376,
    output_width=2688,
  )
  assert_eq(plan.scratch_bytes, 64512L)
  assert_eq(plan.output_bytes, 32256L)
  try
    TimestepMlpLayout::new(
      rows=2147483647,
      input_width=256,
      hidden_width=5376,
      output_width=2688,
    )
  catch {
    @device.DeviceError::InvalidArgument(AotKernelLaunch) => ()
    _ => fail("unexpected extent error")
  } noraise {
    _ => fail("overflow accepted")
  }
}
