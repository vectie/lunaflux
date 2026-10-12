///|
test "independent router laws reach hash and learned decoder layers" {
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
    routing_projection_law=GroupedVectorFloat32(
      groups=1,
      partitions=32,
      vector_width=1,
      publication_partitions=1,
      multiply_add=FusedMultiplyAdd,
    ),
    routing_softplus_law=ThresholdedFloat32(threshold=20.0),
    routing_sum_law=ButterflyFloat32,
  )
  defer selected.close()
  let text = @utf8.decode(selected.source_bytes())
  assert_eq(
    text.split("sqrtf(((x) > (20.0f) ? (x) : (log1pf(expf(x)))))").count(),
    5,
  )
  assert_eq(text.split("selected_partial[8]").count(), 5)
  for layer in 0..<4 {
    assert_true(text.contains("deepseek_l\{layer}_router_project"))
  }
}
