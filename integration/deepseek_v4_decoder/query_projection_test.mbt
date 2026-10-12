///|
test "independent raw query projection reaches window and compressed owners" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let selected = @deepseek_v4_decoder.DeepSeekDecoderSlice::new(
    spec,
    first_layer=0,
    end_layer=4,
    maximum_rows=3,
    maximum_context=256,
    maximum_layer_state_bytes=16777216L,
    maximum_layer_workspace_bytes=16777216L,
    maximum_layer_expert_bytes=8589934592L,
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
    projection_scaled_dot_law=UnscaledBlockMatrixFloat32(
      block_columns=128,
      publication=SeparateMultiplyAdd,
    ),
  )
  defer selected.close()
  let text = @utf8.decode(selected.source_bytes())
  for layer in 0..<4 {
    for stage in ["q_a", "q_b", "kv"] {
      assert_true(
        text.contains("deepseek_l\{layer}_query_kv_\{stage}_block_dot<128,"),
      )
    }
  }
  // Independent policy does not alter output-B or shared expert choices.
  assert_true(text.contains("lf_block_activation_scale"))
  assert_true(!text.contains("shared_mlp_block_dot"))
}

///|
test "output scaled dot reaches sublayers without changing query policy" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let selected = @deepseek_v4_decoder.DeepSeekDecoderSlice::new(
    spec,
    first_layer=0,
    end_layer=4,
    maximum_rows=3,
    maximum_context=256,
    maximum_layer_state_bytes=16777216L,
    maximum_layer_workspace_bytes=16777216L,
    maximum_layer_expert_bytes=8589934592L,
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
    output_scaled_dot_law=UnscaledBlockMatrixFloat32(
      block_columns=128,
      publication=SeparateMultiplyAdd,
    ),
  )
  defer selected.close()
  let text = @utf8.decode(selected.source_bytes())
  for layer in 0..<4 {
    assert_true(
      text.contains("deepseek_l\{layer}_output_output_block_dot<128,"),
    )
    assert_true(
      !text.contains("deepseek_l\{layer}_query_kv_q_a_block_dot<128,"),
    )
  }
}
