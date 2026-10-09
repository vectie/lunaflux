///|
test "plain BF16 context projection has no modulation expansion" {
  let layout = @joint_diffusion_device.ParameterProjectionLayout::new(
    rows=3,
    input_width=5120,
    width=5376,
    groups=1,
    fields=1,
  )
  assert_eq(layout.input_bytes, 30720L)
  assert_eq(layout.weight_bytes, 55050240L)
  assert_eq(layout.bias_bytes, 10752L)
  assert_eq(layout.output_bytes, 32256L)
  try
    @joint_diffusion_device.ParameterProjectionLayout::new(
      rows=0,
      input_width=5120,
      width=5376,
      groups=1,
      fields=1,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("accepted empty context")
  }
}
