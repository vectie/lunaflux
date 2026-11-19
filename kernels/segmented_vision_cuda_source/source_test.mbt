///|
test "attention source carries actual head72 and bounded shared scores" {
  let plan = @precision.SegmentedAttentionPrecision::new(
    maximum_rows=8192,
    maximum_segment=2048,
    head_count=16,
    head_width=72,
  )
  let source = @segmented_vision_cuda_source.attention_source(
    plan,
    symbol="vision_attention",
  )
  assert_eq(source.launch(), (8192, 16, 128))
  assert_eq(source.shared_bytes(), 8704)
  assert_eq(source.parameter_count(), 7)
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("scores[2048]"))
  assert_true(text.contains("d<72"))
  assert_true(text.contains("__float2bfloat16_rn"))
  assert_true(!text.contains("causal"))
  assert_true(
    (try? @segmented_vision_cuda_source.attention_source(
      plan,
      symbol="bad-name",
    ))
    is Err(_),
  )
  let oversized = @precision.SegmentedAttentionPrecision::new(
    maximum_rows=9000,
    maximum_segment=9000,
    head_count=16,
    head_width=72,
  )
  assert_true(
    (try? @segmented_vision_cuda_source.attention_source(
      oversized,
      symbol="wide",
    ))
    is Err(_),
  )
}

///|
test "patch lookup permutation and merger primitives emit bounded dynamic ABIs" {
  let source = @utf8.decode(
    @segmented_vision_cuda_source.patch_rows_source(
      maximum_rows=8192,
      channels=3,
      patch_size=14,
      symbol="patch_rows",
    ),
  )
  assert_true(source.contains("i>=live*588"))
  assert_true(source.contains("uint16_t"))
  assert_true(
    @segmented_vision_cuda_source.position_add_source(
      maximum_rows=8192,
      width=1152,
      symbol="absolute_position",
    ).length() >
    0,
  )
  assert_true(
    @segmented_vision_cuda_source.gather_source(
      maximum_rows=8192,
      width=1152,
      symbol="window_gather",
    ).length() >
    0,
  )
  assert_true(
    @segmented_vision_cuda_source.merge_residual_source(
      maximum_rows=8192,
      width=1152,
      symbol="merger_residual",
    ).length() >
    0,
  )
  assert_true(
    (try? @segmented_vision_cuda_source.patch_rows_source(
      maximum_rows=8192,
      channels=0,
      patch_size=14,
      symbol="invalid",
    ))
    is Err(_),
  )
  assert_true(
    (try? @segmented_vision_cuda_source.merge_residual_source(
      maximum_rows=7,
      width=1152,
      symbol="invalid",
    ))
    is Err(_),
  )
}
