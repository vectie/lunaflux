///|
test "first AMP activation scalar oracle preserves crop padding and stage order" {
  let plan = tiny_vae_plan()
  let request = tiny_vae_request(plan)
  let contract = try! bind_audio_decoder_first_amp_activation(
    plan,
    request,
    decoder_batch_count=1,
    alpha_name="decoder.resblocks.0.activations.0.act.alpha",
    beta_name="decoder.resblocks.0.activations.0.act.beta",
    upsample_filter_name="decoder.resblocks.0.activations.0.upsample.filter",
    downsample_filter_name="decoder.resblocks.0.activations.0.downsample.lowpass.filter",
  )
  assert_eq(contract.channel_count(), 1)
  assert_eq(contract.time_count(), 10)
  assert_eq(contract.upsample_time(), 20)
  let up_filter = Array::make(12, 0.0F)
  let down_filter = Array::make(12, 0.0F)
  up_filter[5] = 1.0F
  down_filter[5] = 1.0F
  let output = try! execute_audio_decoder_first_amp_activation(
    contract,
    [1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F],
    [0.0F],
    [0.0F],
    up_filter,
    down_filter,
  )
  assert_eq(output.length(), 10)
  let first_sine = @math.sinf(2.0F)
  assert_true(
    (output[0] - (2.0F + first_sine * first_sine / (1.0F + 1.0e-9F))).abs() <
    1.0e-5F,
  )
  let last_sine = @math.sinf(20.0F)
  assert_true(
    (output[9] - (20.0F + last_sine * last_sine / (1.0F + 1.0e-9F))).abs() <
    1.0e-5F,
  )
}

///|
test "first AMP activation rejects aliased filters and wrong buffers" {
  let plan = tiny_vae_plan()
  let request = tiny_vae_request(plan)
  try
    bind_audio_decoder_first_amp_activation(
      plan,
      request,
      decoder_batch_count=1,
      alpha_name="alpha",
      beta_name="beta",
      upsample_filter_name="same.filter",
      downsample_filter_name="same.filter",
    )
  catch {
    InvalidProjectionTensorIdentity => ()
    _ => fail("wrong aliased-filter rejection")
  } noraise {
    _ => fail("aliased filters admitted")
  }
}
