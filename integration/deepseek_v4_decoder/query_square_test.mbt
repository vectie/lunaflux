///|
test "independent query fold reaches window and compressed decoder owners" {
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
    query_square_law=VectorizedTreeFloat32(partitions=128, vector_width=4),
  )
  defer selected.close()
  let text = @utf8.decode(selected.source_bytes())
  assert_eq(text.split("float sums[4]").count(), 5)
  assert_eq(text.split("blockDim.x != 128").count(), 5)
  for layer in 0..<4 {
    assert_true(text.contains("deepseek_l\{layer}_query_kv_q_norm"))
  }
}
