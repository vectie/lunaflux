///|
test "staged MLP layout sizes scratch and up-gate-down weights independently" {
  let plan = StagedMlpLayout::new(rows=3, hidden_width=4, intermediate_width=7)
  assert_eq(plan.input_bytes, 24L)
  assert_eq(plan.product_bytes, 42L)
  assert_eq(plan.packed_weight_bytes, 112L)
  assert_eq(plan.down_weight_bytes, 56L)
  for
    (rows, hidden, intermediate) in [
      (0, 4, 7),
      (1, 0, 7),
      (1, 4, 0),
      (2147483647, 1, 2),
      (1, 1, 1073741824),
      (1, 2147483647, 1073741823),
    ] {
    try
      StagedMlpLayout::new(
        rows~,
        hidden_width=hidden,
        intermediate_width=intermediate,
      )
    catch {
      @device.DeviceError::InvalidArgument(AotKernelLaunch) => ()
      _ => fail("wrong MLP shape error")
    } noraise {
      _ => fail("invalid MLP shape accepted")
    }
  }
}
