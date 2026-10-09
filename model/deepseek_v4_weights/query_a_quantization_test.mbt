///|
test "Query-A is replicated raw block-FP8 for all profile geometries" {
  for
    entry in [
      ("deepseek-ai/DeepSeek-V4-Flash", [1024L, 4096L], [8L, 32L]),
      ("deepseek-ai/DeepSeek-V4-Flash-Base", [1024L, 4096L], [8L, 32L]),
      ("deepseek-ai/DeepSeek-V4-Pro", [1536L, 7168L], [12L, 56L]),
      ("deepseek-ai/DeepSeek-V4-Pro-Base", [1536L, 7168L], [12L, 56L]),
      ("deepseek-ai/DeepSeek-V4-Flash-0731", [1024L, 4096L], [8L, 32L]),
    ] {
    let spec = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
      entry.0,
    )
    let layout = query_a_quantized_layout(spec)
    assert_eq(layout.contract(), Fp8E4M3Ue8M0Block128ReplicatedRowMajorV1)
    assert_eq(layout.weight_shape(), ReadOnlyArray::from_array(entry.1))
    assert_eq(layout.scale_shape(), ReadOnlyArray::from_array(entry.2))
    assert_eq(layout.block_width(), 128)
    assert_true(layout.tensor_parallel_replicated())
  }
}
