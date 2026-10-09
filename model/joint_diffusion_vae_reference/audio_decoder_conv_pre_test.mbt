///|
test "audio decoder conv_pre scalar oracle preserves weight norm and B-C-T order" {
  let plan = tiny_vae_plan()
  let request = tiny_vae_request(plan)
  let contract = try! bind_audio_decoder_conv_pre(
    plan,
    request,
    decoder_batch_count=1,
    weight_g_name="decoder.conv_pre.weight_g",
    weight_v_name="decoder.conv_pre.weight_v",
    bias_name="decoder.conv_pre.bias",
  )
  assert_eq(contract.input_element_count(), 8L)
  assert_eq(contract.weight_g_element_count(), 2L)
  assert_eq(contract.weight_v_element_count(), 56L)
  assert_eq(contract.output_element_count(), 4L)
  let weight_v = Array::make(56, 0.0F)
  weight_v[3] = 1.0F
  weight_v[28 + 7 + 3] = 2.0F
  let output = try! execute_audio_decoder_conv_pre(
    contract,
    [1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F],
    [1.0F, 4.0F],
    weight_v,
    [0.5F, -1.0F],
  )
  assert_eq(output, [1.5F, 2.5F, 11.0F, 15.0F])
}

///|
test "audio decoder conv_pre rejects hostile identity length and zero norm" {
  let plan = tiny_vae_plan()
  let request = tiny_vae_request(plan)
  try
    bind_audio_decoder_conv_pre(
      plan,
      request,
      decoder_batch_count=1,
      weight_g_name="../weight_g",
      weight_v_name="decoder.conv_pre.weight_v",
      bias_name="decoder.conv_pre.bias",
    )
  catch {
    InvalidProjectionTensorIdentity => ()
    _ => fail("wrong tensor identity rejection")
  } noraise {
    _ => fail("hostile tensor name admitted")
  }
  let contract = try! bind_audio_decoder_conv_pre(
    plan,
    request,
    decoder_batch_count=1,
    weight_g_name="decoder.conv_pre.weight_g",
    weight_v_name="decoder.conv_pre.weight_v",
    bias_name="decoder.conv_pre.bias",
  )
  try
    execute_audio_decoder_conv_pre(
      contract,
      [1.0F],
      [1.0F, 1.0F],
      Array::make(56, 0.0F),
      [0.0F, 0.0F],
    )
  catch {
    InvalidProjectionInput => ()
    _ => fail("wrong short-input rejection")
  } noraise {
    _ => fail("short input admitted")
  }
  try
    execute_audio_decoder_conv_pre(
      contract,
      Array::make(8, 1.0F),
      [1.0F, 1.0F],
      Array::make(56, 0.0F),
      [0.0F, 0.0F],
    )
  catch {
    InvalidProjectionInput => ()
    _ => fail("wrong zero-norm rejection")
  } noraise {
    _ => fail("zero weight_v norm admitted")
  }
}
