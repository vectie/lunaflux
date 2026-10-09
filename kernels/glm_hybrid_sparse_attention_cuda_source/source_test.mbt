///|
test "selected-index sparse attention source fixes gather and numeric semantics" {
  let source = selected_index_sparse_attention_bf16_source(
    query_head_count=64,
    key_value_head_count=64,
    query_key_dimension=256,
    value_dimension=256,
    selected_width=2051,
    maximum_tokens=32,
    function_symbol="lunaflux_glm_hybrid_dsa_sparse_attention_bf16_v1",
  )
  assert_eq(source.operation(), SelectedIndexSparseAttentionBf16)
  assert_eq(source.selected_width(), 2051)
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("const int kv_head = head / groups;"))
  assert_true(text.contains("key < 0 || key >= keys_live || key > position"))
  assert_true(text.contains("if (key == -1) continue;"))
  assert_true(text.contains("row_valid = 0; break;"))
  assert_true(text.contains("if (!row_valid) continue;"))
  assert_true(text.contains("__float2bfloat16_rn(__fdividef"))
  assert_false(text.contains("topk"))
  assert_false(text.contains("pool_ape"))
}

///|
test "sparse attention renderer rejects invalid GQA and size overflow" {
  try
    selected_index_sparse_attention_bf16_source(
      query_head_count=63,
      key_value_head_count=32,
      query_key_dimension=256,
      value_dimension=256,
      selected_width=2048,
      maximum_tokens=32,
      function_symbol="lunaflux_glm_hybrid_dsa_sparse_attention_bf16_v1",
    )
  catch {
    InvalidKeyValueHeadCount => ()
    _ => fail("unexpected invalid-GQA error")
  } noraise {
    _ => fail("nondivisible GQA admitted")
  }
  try
    selected_index_sparse_attention_bf16_source(
      query_head_count=1024,
      key_value_head_count=1,
      query_key_dimension=65536,
      value_dimension=65536,
      selected_width=65536,
      maximum_tokens=1048576,
      function_symbol="lunaflux_glm_hybrid_dsa_sparse_attention_bf16_v1",
    )
  catch {
    SizeOverflow => ()
    _ => fail("unexpected overflow error")
  } noraise {
    _ => fail("overflowing sparse-attention envelope admitted")
  }
}
