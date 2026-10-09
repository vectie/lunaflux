///|
test "complete AMP scratch keeps all parallel results alive" {
  let plan = @joint_diffusion_device.AudioAmpLayout::new(
    batch=2,
    channels=8,
    time=31,
    max_bytes=100000L,
  )
  assert_eq(plan.value_offsets.length(), 7)
  assert_eq(plan.convolutions.length(), 3)
  assert_true(plan.value_offsets[0] >= plan.activation.scratch_bytes)
  for i in 1..<7 {
    assert_true(
      plan.value_offsets[i] >=
      plan.value_offsets[i - 1] + plan.activation.value_bytes,
    )
  }
  assert_true(
    plan.normalized_offset >=
    plan.value_offsets[6] + plan.activation.value_bytes,
  )
  assert_eq(plan.convolutions[2].weight_bytes, 8L * 8L * 11L * 4L)
  ignore(
    @joint_diffusion_device.AudioAmpLayout::new(
      batch=2,
      channels=8,
      time=31,
      max_bytes=plan.scratch_bytes,
    ),
  )
  try
    @joint_diffusion_device.AudioAmpLayout::new(
      batch=2,
      channels=8,
      time=31,
      max_bytes=plan.scratch_bytes - 1L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("short AMP scratch accepted")
  }
}
