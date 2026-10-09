///|
test "generic AudioVAE upsample transition preserves per-input normalization and ConvTranspose order" {
  let plan = tiny_vae_plan()
  let request = tiny_vae_request(plan)
  let contract = try! bind_audio_decoder_upsample_transition(
    plan,
    request,
    stage_index=1,
    decoder_batch_count=1,
    input_channels=2,
    output_channels=1,
    input_time=2,
    output_time=3,
    kernel_size=3,
    stride=2,
    padding=1,
    weight_g_name="decoder.ups.1.0.weight_g",
    weight_v_name="decoder.ups.1.0.weight_v",
    bias_name="decoder.ups.1.0.bias",
  )
  assert_eq(contract.axes(), PerInputChannelOutputAndKernel)
  assert_eq(contract.input_element_count(), 4L)
  assert_eq(contract.weight_g_element_count(), 2L)
  assert_eq(contract.weight_v_element_count(), 6L)
  assert_eq(contract.output_element_count(), 3L)
  let output = try! execute_audio_decoder_upsample_transition(
    contract,
    [1.0F, 2.0F, 3.0F, 4.0F],
    [1.0F, 1.0F],
    [0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F],
    [0.0F],
  )
  assert_eq(output, [4.0F, 0.0F, 6.0F])
}

///|
test "generic AudioVAE upsample transition rejects invalid output formula identity and values" {
  let plan = tiny_vae_plan()
  let request = tiny_vae_request(plan)
  try
    bind_audio_decoder_upsample_transition(
      plan,
      request,
      stage_index=1,
      decoder_batch_count=1,
      input_channels=2,
      output_channels=1,
      input_time=1,
      output_time=1,
      kernel_size=1,
      stride=1,
      padding=1,
      weight_g_name="decoder.ups.1.0.weight_g",
      weight_v_name="decoder.ups.1.0.weight_v",
      bias_name="decoder.ups.1.0.bias",
    )
  catch {
    GeometryOverflow => ()
    _ => fail("wrong invalid-output-extent rejection")
  } noraise {
    _ => fail("non-positive output extent admitted")
  }
  let contract = try! bind_audio_decoder_upsample_transition(
    plan,
    request,
    stage_index=1,
    decoder_batch_count=1,
    input_channels=2,
    output_channels=1,
    input_time=2,
    output_time=3,
    kernel_size=3,
    stride=2,
    padding=1,
    weight_g_name="decoder.ups.1.0.weight_g",
    weight_v_name="decoder.ups.1.0.weight_v",
    bias_name="decoder.ups.1.0.bias",
  )
  try
    execute_audio_decoder_upsample_transition(
      contract,
      Array::make(4, 1.0F),
      [1.0F, 1.0F],
      Array::make(6, 0.0F),
      [0.0F],
    )
  catch {
    InvalidProjectionInput => ()
    _ => fail("wrong zero-norm rejection")
  } noraise {
    _ => fail("zero weight-v norm admitted")
  }
}
