///|
test "full transformer stack requires every layer exactly once in order" {
  validate_layer_order(Array::makei(50, i => i), 50)
  for invalid in [[0, 2, 1], [0, 1, 1], [0, 1], [1, 2, 3]] {
    try validate_layer_order(invalid, 3) catch {
      @device.DeviceError::InvalidArgument(AotKernelLaunch) => ()
      _ => fail("wrong layer order error")
    } noraise {
      _ => fail("invalid stack accepted")
    }
  }
}
