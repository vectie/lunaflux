///|
test "device accepts inclusive memory and identifier boundaries" {
  let low = @device.new([0], 268435456L)
  assert_eq(low.visible_devices()[0], 0)
  assert_eq(low.memory_ceiling_bytes(), 268435456L)
  let high = @device.new([1023], 1125899906842624L)
  assert_eq(high.visible_devices()[0], 1023)
  assert_eq(high.memory_ceiling_bytes(), 1125899906842624L)
}

///|
test "device rejects duplicate identity and too-small memory" {
  try @device.new([2, 2], 1073741824L) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=DeviceId, issue=Duplicate, index=1)",
      )
  } noraise {
    _ => fail("expected duplicate device rejection")
  }
  try @device.new([0], 268435455L) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=MemoryCeilingBytes, issue=BelowMinimum, index=0)",
      )
  } noraise {
    _ => fail("expected memory floor rejection")
  }
}
