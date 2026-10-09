///|
test "all five profiles bind replicated 512-wide key/value storage" {
  for
    entry in [
      ("deepseek-ai/DeepSeek-V4-Flash", 4096L),
      ("deepseek-ai/DeepSeek-V4-Flash-Base", 4096L),
      ("deepseek-ai/DeepSeek-V4-Pro", 7168L),
      ("deepseek-ai/DeepSeek-V4-Pro-Base", 7168L),
      ("deepseek-ai/DeepSeek-V4-Flash-0731", 4096L),
    ] {
    let layout = key_value_quantized_layout(
      @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(entry.0),
    )
    assert_eq(layout.contract(), Fp8E4M3Ue8M0Block128ReplicatedRowMajorV1)
    assert_eq(layout.weight_shape(), [512L, entry.1])
    assert_eq(layout.scale_shape(), [4L, entry.1 / 128L])
    assert_eq(layout.input_width().to_int64(), entry.1)
    assert_eq(layout.output_width(), 512)
    assert_eq(layout.block_width(), 128)
    assert_true(layout.tensor_parallel_replicated())
  }
}
