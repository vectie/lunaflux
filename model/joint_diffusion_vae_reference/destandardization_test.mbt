///|
test "video latent destandardization binds request identity and staged F32 arithmetic" {
  let plan = tiny_vae_plan()
  let request = tiny_vae_request(plan)
  let contract = bind_video_latent_destandardization(
    plan,
    request,
    means=[0.5F, -1.0F],
    standard_deviations=[2.0F, 0.25F],
  )
  assert_eq(contract.plan_digest(), plan.plan_digest())
  assert_eq(contract.request_shape(), request)
  assert_eq(contract.storage(), Float32InputScaleMeanOutput)
  assert_eq(contract.layout(), BatchChannelFrameHeightWidth)
  assert_eq(contract.channel_count(), 2)
  assert_eq(contract.batch_count(), 1)
  assert_eq(contract.elements_per_channel(), 12L)
  assert_eq(contract.element_count(), 24L)
  assert_eq(contract.mean(0), Some(0.5F))
  assert_eq(contract.standard_deviation(1), Some(0.25F))
  assert_eq(contract.mean(2), None)
  let input = Array::make(24, 1.0F)
  let output = destandardize_video_latents(contract, input)
  assert_eq(output[:12], Array::make(12, 2.5F))
  assert_eq(output[12:], Array::make(12, -0.75F))
  assert_eq(
    contract.digest(),
    bind_video_latent_destandardization(plan, request, means=[0.5F, -1.0F], standard_deviations=[
      2.0F, 0.25F,
    ]).digest(),
  )
}

///|
test "video latent destandardization rejects hostile identity coefficients and payload" {
  let plan = tiny_vae_plan()
  let request = tiny_vae_request(plan)
  // Request shapes carry geometry, not provenance. An equal shape reconstructed
  // by another plan is valid geometry for this plan; the resulting contract is
  // nevertheless bound to the explicitly supplied plan digest.
  let geometry_compatible = tiny_vae_plan(seed='8')
  let compatible = bind_video_latent_destandardization(
    plan,
    tiny_vae_request(geometry_compatible),
    means=[0.0F, 0.0F],
    standard_deviations=[1.0F, 1.0F],
  )
  assert_eq(compatible.plan_digest(), plan.plan_digest())
  for
    hostile in [
      ([0.0F], [1.0F, 1.0F]),
      ([0.0F, 0.0F], [1.0F, 0.0F]),
      ([0.0F, Float::reinterpret_from_uint(0x7f800000U)], [1.0F, 1.0F]),
    ] {
    let (means, deviations) = hostile
    try
      bind_video_latent_destandardization(
        plan,
        request,
        means~,
        standard_deviations=deviations,
      )
    catch {
      InvalidDestandardizationCoefficients => ()
      _ => fail("unexpected coefficient rejection")
    } noraise {
      _ => fail("hostile coefficients admitted")
    }
  }
  let contract = bind_video_latent_destandardization(
    plan,
    request,
    means=[0.0F, 0.0F],
    standard_deviations=[1.0F, 1.0F],
  )
  try destandardize_video_latents(contract, Array::make(23, 0.0F)) catch {
    InvalidDestandardizationInput => ()
    _ => fail("unexpected short-input rejection")
  } noraise {
    _ => fail("short normalized latent admitted")
  }
  let nonfinite = Array::make(24, 0.0F)
  nonfinite[3] = Float::reinterpret_from_uint(0x7fc00000U)
  try destandardize_video_latents(contract, nonfinite) catch {
    InvalidDestandardizationInput => ()
    _ => fail("unexpected nonfinite-input rejection")
  } noraise {
    _ => fail("nonfinite normalized latent admitted")
  }
}
