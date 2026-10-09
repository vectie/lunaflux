///|
test "conditioned block derives distinct timestep rows without truncating modalities" {
  assert_eq(conditioning_rows(12, 3), 4)
  assert_eq(conditioning_rows(3, 3), 1)
  assert_eq(conditioning_rows(4, 1), 4)
  for (rows, groups) in [(0, 3), (4, 3), (12, 0), (-3, 3)] {
    try conditioning_rows(rows, groups) catch {
      @device.DeviceError::InvalidArgument(AotKernelLaunch) => ()
      _ => fail("wrong conditioning geometry error")
    } noraise {
      _ => fail("incomplete modality group accepted")
    }
  }
}
