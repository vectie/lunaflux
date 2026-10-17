///|
test "score reuse keeps ordered arithmetic and distinct ownership" {
  let source = @selected_sparse_attention_cuda_source.selected_sparse_source(
    query_heads=64,
    kv_heads=64,
    qk_dimension=256,
    value_dimension=256,
    selected_width=2051,
    maximum_queries=1,
    maximum_history=64,
    symbol="selected",
  )
  assert_eq(source.launch(), (64, 256))
  assert_eq(source.scratch_bytes(), 32768L)
  let text = @utf8.decode(source.source_bytes())
  assert_false(text.contains("blockIdx.x != 0"))
  assert_eq(text.split("float dot = 0.0f;").count(), 2)
  assert_true(text.contains("key == -1"))
  assert_true(text.contains("key > position"))
  assert_true(text.contains("__syncthreads(); // score readers retire"))
  assert_true(text.contains("__float2bfloat16_rn(__fdividef"))
  assert_false(text.contains("__expf"))
  assert_false(text.contains("__shfl"))
  assert_true(text.contains("rank < 2051"))
}

///|
test "producer prefix changes traversal but not index stride or arithmetic" {
  for extent in [0, 3, 67, 2051] {
    let source = @selected_sparse_attention_cuda_source.selected_sparse_source(
      query_heads=2,
      kv_heads=1,
      qk_dimension=4,
      value_dimension=4,
      selected_width=2051,
      maximum_queries=1,
      maximum_history=64,
      symbol="bound",
      published_slots=extent,
    )
    let text = @utf8.decode(source.source_bytes())
    assert_true(text.contains("rank < " + extent.to_string()))
    assert_true(text.contains("storage stride remains 2051"))
    assert_true(text.contains("key > position"))
    assert_false(text.contains("__expf"))
  }
}

///|
test "unsupported source geometry is rejected before scratch allocation" {
  try
    @selected_sparse_attention_cuda_source.selected_sparse_source(
      query_heads=3,
      kv_heads=2,
      qk_dimension=256,
      value_dimension=256,
      selected_width=2051,
      maximum_queries=1,
      maximum_history=64,
      symbol="selected",
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("invalid GQA admitted")
  }
}
