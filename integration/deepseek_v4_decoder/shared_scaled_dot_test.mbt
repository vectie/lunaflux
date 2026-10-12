///|
test "shared scaled-dot policy reaches decoder slice without changing routed FP4 or defaults" {
  let spec = @spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash",
  )
  let law : @precision.ScaledDotReductionLaw = UnscaledBlockMatrixFloat32(
    block_columns=128,
    publication=SeparateMultiplyAdd,
  )
  let selected = @deepseek_v4_decoder.DeepSeekDecoderSlice::new(
    spec,
    first_layer=0,
    end_layer=1,
    maximum_rows=1,
    maximum_context=256,
    maximum_layer_state_bytes=8589934592L,
    maximum_layer_workspace_bytes=8589934592L,
    maximum_layer_expert_bytes=8589934592L,
    accuracy=@precision.Accuracy::new(absolute=0.02, relative=0.02),
    shared_scaled_dot_law=law,
  )
  defer selected.close()
  let text = @utf8.decode(selected.source_bytes())
  assert_true(text.contains("deepseek_l0_shared_block_dot<4096>"))
  assert_true(text.contains("deepseek_l0_shared_block_dot<2048>"))
  assert_true(text.contains("deepseek_l0_routed_a0"))
  assert_false(text.contains("deepseek_l0_routed_block_dot"))
}
