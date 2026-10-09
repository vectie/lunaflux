///|
test "alias-free activation checks simultaneous intermediate storage" {
  let layout = @joint_diffusion_device.AliasFreeActivationLayout::new(
    batch=2,
    channels=512,
    time=1035,
    max_scratch_bytes=20000000L,
  )
  assert_eq(layout.value_bytes, 4239360L)
  assert_eq(layout.intermediate_bytes, 8478720L)
  assert_eq(layout.parameter_bytes, 2048L)
  assert_true(layout.activated_offset >= layout.intermediate_bytes)
  assert_eq(layout.activated_offset % 256L, 0L)
  ignore(
    @joint_diffusion_device.AliasFreeActivationLayout::new(
      batch=2,
      channels=512,
      time=1035,
      max_scratch_bytes=layout.scratch_bytes,
    ),
  )
  for
    (batch, channels, time, budget) in [
      (2, 512, 1035, layout.scratch_bytes - 1L),
      (0, 512, 1035, 20000000L),
      (2, 512, 1073741824, 9223372036854775807L),
      (2147483647, 2147483647, 2, 9223372036854775807L),
    ] {
    try
      @joint_diffusion_device.AliasFreeActivationLayout::new(
        batch~,
        channels~,
        time~,
        max_scratch_bytes=budget,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("invalid alias-free geometry accepted")
    }
  }
}
