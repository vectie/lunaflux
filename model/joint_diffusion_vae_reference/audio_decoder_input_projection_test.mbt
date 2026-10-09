///|
test "audio decoder input projection scalar oracle preserves B-C-T layout" {
  let plan = tiny_vae_plan()
  let request = tiny_vae_request(plan)
  let contract = try! bind_audio_decoder_input_projection(
    plan,
    request,
    decoder_batch_count=1,
    weight_name="dec_in_proj.weight",
    bias_name="dec_in_proj.bias",
  )
  assert_eq(contract.input_element_count(), 4L)
  assert_eq(contract.weight_element_count(), 8L)
  assert_eq(contract.output_element_count(), 8L)
  let output = try! project_audio_decoder_input(
    contract,
    [1.0F, 2.0F, 3.0F, 4.0F],
    [1.0F, 10.0F, 2.0F, 20.0F, 3.0F, 30.0F, 4.0F, 40.0F],
    [0.5F, 1.0F, 1.5F, 2.0F],
  )
  assert_eq(output, [31.5F, 42.5F, 63.0F, 85.0F, 94.5F, 127.5F, 126.0F, 170.0F])
  try
    bind_audio_decoder_input_projection(
      plan,
      request,
      decoder_batch_count=1,
      weight_name="../weight",
      bias_name="dec_in_proj.bias",
    )
  catch {
    InvalidProjectionTensorIdentity => ()
    _ => fail("wrong tensor identity rejection")
  } noraise {
    _ => fail("hostile tensor name admitted")
  }
  try
    project_audio_decoder_input(
      contract,
      [1.0F],
      [1.0F, 10.0F, 2.0F, 20.0F, 3.0F, 30.0F, 4.0F, 40.0F],
      [0.5F, 1.0F, 1.5F, 2.0F],
    )
  catch {
    InvalidProjectionInput => ()
    _ => fail("wrong projection input rejection")
  } noraise {
    _ => fail("short projection input admitted")
  }
}
