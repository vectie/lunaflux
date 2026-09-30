///|
test "contraction is explicit through semantics schedule source and ABI" {
  let lowering = source_test_decode_lowering(
    candidate_id=456,
    alternative=true,
    key_value_heads=8,
    tokens_per_page=8,
  )
  let abi = CudaPagedAttentionTileAbi::new(
    query_heads=16,
    key_value_heads=8,
    head_dimension=128,
    tokens_per_page=8,
    input_row_width=4096,
    page_stride_values=8192L,
    physical_page_capacity=16384L,
    maximum_query_rows=32,
    maximum_query_tokens=2048,
    maximum_page_entries=16384,
  )
  let source = emit_cuda_attention_tile_source(
    lowering, abi, "explicit", 1048576,
  )
  assert_eq(source.symbol(), "explicit_blockwise_fma_f32_v3")
  let text = @utf8.decode(source.bytes())
  assert_true(text.contains("numeric_law=blockwise-fma-f32-probability-v3"))
  assert_true(text.contains("dot = fmaf("))
  assert_true(text.contains("tile_numerator[item] = fmaf("))
  assert_true(text.contains("accumulator[item] = fmaf("))
  assert_true(text.contains("expf(scores[key] - tile_maximum)"))
  assert_false(text.contains("__expf("))
  assert_false(text.contains("__exp2f("))
  let strict = emit_cuda_attention_tile_source(
    source_test_decode_lowering(
      candidate_id=452,
      alternative=true,
      key_value_heads=8,
      tokens_per_page=8,
    ),
    abi,
    "strict",
    1048576,
  )
  assert_false(@utf8.decode(strict.bytes()).contains("fmaf("))
}
