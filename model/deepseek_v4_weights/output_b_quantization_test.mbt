///|
test "Flash Output-B keeps raw block-FP8 row-major checkpoint layout" {
  let layout = output_b_quantized_layout(
    @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
      "deepseek-ai/DeepSeek-V4-Flash-0731",
    ),
  )
  assert_eq(layout.contract(), Fp8E4M3Ue8M0Block128RowMajorV1)
  assert_eq(layout.weight_shape(), [4096L, 8192L])
  assert_eq(layout.scale_shape(), [32L, 64L])
  assert_eq(layout.input_width(), 8192)
  assert_eq(layout.output_width(), 4096)
  assert_eq(layout.block_width(), 128)
  assert_eq(layout.tensor_parallel_shard_axis(), 1)
}

///|
test "Pro Output-B derives its distinct official dimensions" {
  let layout = output_b_quantized_layout(
    @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
      "deepseek-ai/DeepSeek-V4-Pro",
    ),
  )
  assert_eq(layout.weight_shape(), [7168L, 16384L])
  assert_eq(layout.scale_shape(), [56L, 128L])
}
