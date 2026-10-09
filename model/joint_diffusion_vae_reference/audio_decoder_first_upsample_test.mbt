///|
test "first AudioVAE upsampler scalar oracle preserves transposed-convolution order" {
  let plan = tiny_vae_plan()
  let request = tiny_vae_request(plan)
  let contract = try! bind_audio_decoder_first_upsample(
    plan,
    request,
    decoder_batch_count=1,
    weight_g_name="decoder.ups.0.0.weight_g",
    weight_v_name="decoder.ups.0.0.weight_v",
    bias_name="decoder.ups.0.0.bias",
  )
  assert_eq(contract.input_element_count(), 4L)
  assert_eq(contract.weight_g_element_count(), 2L)
  assert_eq(contract.weight_v_element_count(), 18L)
  assert_eq(contract.output_time(), 10)
  assert_eq(contract.output_element_count(), 10L)
  let weight_v = Array::make(18, 0.0F)
  weight_v[2] = 1.0F
  weight_v[9 + 2] = 1.0F
  let output = try! execute_audio_decoder_first_upsample(
    contract,
    [1.0F, 2.0F, 3.0F, 4.0F],
    [1.0F, 1.0F],
    weight_v,
    [0.0F],
  )
  assert_eq(output, [4.0F, 0.0F, 0.0F, 0.0F, 0.0F, 6.0F, 0.0F, 0.0F, 0.0F, 0.0F])
}

///|
test "first AudioVAE upsampler rejects ConvPre-axis and hostile values by construction" {
  let plan = tiny_vae_plan()
  let request = tiny_vae_request(plan)
  try
    bind_audio_decoder_first_upsample(
      plan,
      request,
      decoder_batch_count=1,
      weight_g_name="../weight_g",
      weight_v_name="decoder.ups.0.0.weight_v",
      bias_name="decoder.ups.0.0.bias",
    )
  catch {
    InvalidProjectionTensorIdentity => ()
    _ => fail("wrong tensor identity rejection")
  } noraise {
    _ => fail("hostile tensor identity admitted")
  }
  let contract = try! bind_audio_decoder_first_upsample(
    plan,
    request,
    decoder_batch_count=1,
    weight_g_name="decoder.ups.0.0.weight_g",
    weight_v_name="decoder.ups.0.0.weight_v",
    bias_name="decoder.ups.0.0.bias",
  )
  assert_eq(contract.axes(), PerInputChannelOutputAndKernel)
  assert_true(contract.axes() != PerOutputChannelInputAndKernel)
  try
    execute_audio_decoder_first_upsample(
      contract,
      Array::make(4, 1.0F),
      [1.0F, 1.0F],
      Array::make(18, 0.0F),
      [0.0F],
    )
  catch {
    InvalidProjectionInput => ()
    _ => fail("wrong zero-norm rejection")
  } noraise {
    _ => fail("zero weight_v norm admitted")
  }
}
