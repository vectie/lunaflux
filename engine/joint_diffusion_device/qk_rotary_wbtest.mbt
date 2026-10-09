///|
test "QK rotary binding geometry preserves head dimensions and bounded tensor extents" {
  let (tensor, vector, _) = qk_rotary_geometry(64, 96, 56, 128)
  assert_eq(tensor, 64L * 56L * 128L * 2L)
  assert_eq(vector, 256L)
  for
    (rows, rotary, heads, width) in [
      (0, 96, 56, 128),
      (64, 97, 56, 128),
      (64, 130, 56, 128),
      (2147483647, 96, 56, 128),
      (64, 96, 0, 128),
    ] {
    try qk_rotary_geometry(rows, rotary, heads, width) catch {
      @device.DeviceError::InvalidArgument(AotKernelLaunch) => ()
      _ => fail("wrong geometry error")
    } noraise {
      _ => fail("invalid rotary geometry admitted")
    }
  }
}
