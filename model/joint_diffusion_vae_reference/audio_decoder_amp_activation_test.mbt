///|
test "generic indexed AMP activation binds exact geometry and shared scalar order" {
  let plan = tiny_vae_plan()
  let request = tiny_vae_request(plan)
  let contract = try! bind_audio_decoder_amp_activation(
    plan,
    request,
    stage_index=1,
    decoder_batch_count=1,
    channel_count=1,
    time_count=2,
    alpha_name="decoder.resblocks.3.activations.0.act.alpha",
    beta_name="decoder.resblocks.3.activations.0.act.beta",
    upsample_filter_name="decoder.resblocks.3.activations.0.upsample.filter",
    downsample_filter_name="decoder.resblocks.3.activations.0.downsample.lowpass.filter",
  )
  assert_eq(contract.stage_index(), 1)
  assert_eq(contract.element_count(), 2L)
  assert_eq(contract.intermediate_element_count(), 4L)
  let output = try! execute_audio_decoder_amp_activation(
    contract,
    [0.0F, 0.0F],
    [0.0F],
    [0.0F],
    Array::make(12, 1.0F),
    Array::make(12, 1.0F),
  )
  assert_eq(output, [0.0F, 0.0F])
}

///|
test "generic indexed AMP activation rejects identity and geometry substitutions" {
  let plan = tiny_vae_plan()
  let request = tiny_vae_request(plan)
  try
    bind_audio_decoder_amp_activation(
      plan,
      request,
      stage_index=-1,
      decoder_batch_count=1,
      channel_count=1,
      time_count=2,
      alpha_name="decoder.resblocks.3.activations.0.act.alpha",
      beta_name="decoder.resblocks.3.activations.0.act.beta",
      upsample_filter_name="same.filter",
      downsample_filter_name="same.filter",
    )
  catch {
    InvalidProjectionTensorIdentity => ()
    _ => fail("wrong AMP activation rejection")
  } noraise {
    _ => fail("hostile AMP activation admitted")
  }
}
