///|
test "matrix schedule requires an explicit numerical alternative" {
  let make = accumulation => {
    @selected_sparse_attention_cuda_source.selected_sparse_source(
      query_heads=4,
      kv_heads=2,
      qk_dimension=32,
      value_dimension=16,
      selected_width=65,
      maximum_queries=8,
      maximum_history=128,
      symbol="sparse_matrix_test",
      accumulation~,
    )
  }
  let ordered = @utf8.decode(make(@precision.OrderedFloat32).source_bytes())
  let matrix = @utf8.decode(make(@precision.TiledFloat32).source_bytes())
  assert_false(ordered.contains("mma.sync"))
  assert_true(matrix.contains("mma.sync.aligned.m16n8k16.row.col.f32.bf16"))
  assert_true(matrix.contains("if(g==0)"))
  assert_true(matrix.contains("key > position"))
  assert_true(matrix.contains("if (!row_valid) return"))
  assert_true(matrix.contains("__float2bfloat16_rn"))
  assert_eq(make(@precision.TiledFloat32).launch(), (32, 256))
  try
    @selected_sparse_attention_cuda_source.selected_sparse_source(
      query_heads=1,
      kv_heads=1,
      qk_dimension=17,
      value_dimension=16,
      selected_width=65,
      maximum_queries=8,
      maximum_history=128,
      symbol="unsupported",
      accumulation=TiledFloat32,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("unsupported matrix tail silently fell back")
  }
}
