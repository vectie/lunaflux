///|
test "official output-A conversion evidence preserves group-major slices" {
  for
    model_id in [
      "deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Flash-Base", "deepseek-ai/DeepSeek-V4-Pro",
      "deepseek-ai/DeepSeek-V4-Pro-Base", "deepseek-ai/DeepSeek-V4-Flash-0731",
    ] {
    let spec = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
      model_id,
    )
    let layout = output_a_converted_layout(spec)
    let groups = spec.output_group_count()
    let input = spec.attention_head_count() * 512 / groups
    assert_eq(layout.source_weight_shape(), [
      (groups * 1024).to_int64(),
      input.to_int64(),
    ])
    assert_eq(layout.source_scale_shape(), [(groups * 8).to_int64(), 32L])
    assert_eq(layout.converted_grouped_shape(), [
      groups.to_int64(),
      1024L,
      input.to_int64(),
    ])
  }
}
