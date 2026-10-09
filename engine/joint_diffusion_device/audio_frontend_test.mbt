///|
test "audio frontend preserves stereo and time with bounded workspace" {
  let plan = @joint_diffusion_device.AudioFrontendLayout::new(
    batch=2,
    input_channels=32,
    hidden_channels=2048,
    output_channels=1024,
    time=207,
    kernel=7,
    max_bytes=100000000L,
  )
  assert_eq(plan.input_bytes, 2L * 32L * 207L * 4L)
  assert_eq(plan.convolution.output_bytes, 2L * 1024L * 207L * 4L)
  assert_eq(plan.projection_weight_bytes, 262144L)
  assert_eq(plan.g_offset, 270336L)
  assert_eq(plan.v_offset, 274432L)
  assert_eq(plan.bias_offset, 58994688L)
  assert_eq(plan.weight_bytes, 58998784L)
  assert_true(plan.normalized_offset >= plan.convolution.value_bytes)
  assert_eq(plan.normalized_offset % 256L, 0L)
  ignore(
    @joint_diffusion_device.AudioFrontendLayout::new(
      batch=2,
      input_channels=32,
      hidden_channels=2048,
      output_channels=1024,
      time=207,
      kernel=7,
      max_bytes=plan.scratch_bytes,
    ),
  )
  try
    @joint_diffusion_device.AudioFrontendLayout::new(
      batch=2,
      input_channels=32,
      hidden_channels=2048,
      output_channels=1024,
      time=207,
      kernel=7,
      max_bytes=plan.scratch_bytes - 1L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("short frontend budget accepted")
  }
}
