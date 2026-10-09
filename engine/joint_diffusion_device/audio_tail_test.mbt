///|
test "waveform tail reduces channels and retains stereo batch" {
  let plan = @joint_diffusion_device.AudioTailLayout::new(
    batch=2,
    channels=8,
    time=165600,
    max_bytes=200000000L,
  )
  assert_eq(plan.convolution.output_bytes, 2L * 165600L * 4L)
  assert_eq(plan.convolution.weight_bytes, 8L * 7L * 4L)
  assert_eq(plan.convolution.channel_bytes, 4L)
  assert_eq(plan.weight_bytes, 388L)
  assert_true(plan.activated_offset >= plan.activation.scratch_bytes)
  assert_true(
    plan.normalized_offset >=
    plan.activated_offset + plan.activation.value_bytes,
  )
  ignore(
    @joint_diffusion_device.AudioTailLayout::new(
      batch=2,
      channels=8,
      time=165600,
      max_bytes=plan.scratch_bytes,
    ),
  )
  try
    @joint_diffusion_device.AudioTailLayout::new(
      batch=2,
      channels=8,
      time=165600,
      max_bytes=plan.scratch_bytes - 1L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("short waveform scratch accepted")
  }
}
