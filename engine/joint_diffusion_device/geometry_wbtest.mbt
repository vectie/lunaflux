///|
test "linear geometry handles partial blocks and Int32 extent without overflow" {
  for block in [1, 32, 128, 256, 1024] {
    for elements in [1L, 31L, 32L, 33L, 13248L, 3580416L, 2147483647L] {
      ignore(linear_dimensions(elements, block))
    }
  }
  for
    (elements, block) in [
      (0L, 256),
      (-1L, 256),
      (2147483648L, 256),
      (1L, 0),
      (1L, -1),
      (1L, 1025),
    ] {
    try linear_dimensions(elements, block) catch {
      @device.DeviceError::InvalidArgument(AotKernelLaunch) => ()
      _ => fail("unexpected geometry error")
    } noraise {
      _ => fail("invalid geometry accepted")
    }
  }
}
