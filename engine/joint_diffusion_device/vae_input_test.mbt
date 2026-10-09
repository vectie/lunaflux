///|
test "stereo VAE input plan preserves latent count and bounds storage" {
  let permutation = @dense_permutation_plan.Permutation::new(
    [2, 207, 32],
    [0, 2, 1],
    max_elements=13248L,
  )
  let layout = @joint_diffusion_device.VaeInputLayout::new(
    permutation,
    batch=2,
    channels=32,
    elements_per_channel=207L,
    max_bytes=52992L,
  )
  assert_eq(layout.value_bytes, 52992L)
  assert_eq(layout.parameter_bytes, 128L)
  try
    @joint_diffusion_device.VaeInputLayout::new(
      permutation,
      batch=2,
      channels=32,
      elements_per_channel=207L,
      max_bytes=52991L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("undersized latent accepted")
  }
  try
    @joint_diffusion_device.VaeInputLayout::new(
      permutation,
      batch=1,
      channels=32,
      elements_per_channel=207L,
      max_bytes=52992L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("mono geometry accepted")
  }
  try
    @joint_diffusion_device.VaeInputLayout::new(
      permutation,
      batch=2,
      channels=32,
      elements_per_channel=9223372036854775807L,
      max_bytes=9223372036854775807L,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("overflow geometry accepted")
  }
}
