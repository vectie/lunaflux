///|
test "transformer block preserves producer consumer ordering without per-block waits" {
  let ordered = compose_block([0, 1], [2, 3, 4, 5], 6, [7, 8], [9, 10], 11)
  assert_eq(ordered.length(), 12)
  for i, stage in ordered {
    assert_eq(stage, i)
  }
  try compose_block([0], [2, 3, 4, 5], 6, [7, 8], [9, 10], 11) catch {
    @device.DeviceError::InvalidArgument(AotKernelLaunch) => ()
    _ => fail("wrong block composition error")
  } noraise {
    _ => fail("missing normalization/modulation stage accepted")
  }
}
