///|
test "projected DSA index source keeps stable selection and tail explicit" {
  let full = projected_index_score_stable_topk_bf16_source(
    head_count=32,
    head_dimension=128,
    top_k=2048,
    pool_size=1,
    append_incomplete_tail=false,
    maximum_tokens=4096,
    function_symbol="lunaflux_glm_hybrid_dsa_index_select_bf16_v1",
  )
  let flash = projected_index_score_stable_topk_bf16_source(
    head_count=32,
    head_dimension=128,
    top_k=2048,
    pool_size=4,
    append_incomplete_tail=true,
    maximum_tokens=4096,
    function_symbol="lunaflux_glm_hybrid_dsa_index_select_bf16_v1",
  )
  assert_eq(full.candidate_capacity(), 4096)
  assert_eq(full.output_width(), 2048)
  assert_eq(flash.candidate_capacity(), 1024)
  assert_eq(flash.output_width(), 2051)
  let source = @utf8.decode(flash.source_bytes())
  assert_true(source.contains("score == best_score"))
  assert_true(source.contains("candidate < best"))
  assert_true(source.contains("final_raw > query_position"))
  assert_true(source.contains("const int tail_count = visible_count % 4;"))
  assert_true(source.contains("const int tail_base = selection_limit * 4;"))
  assert_false(source.contains("const int tail_base = 2048;"))
  assert_false(source.contains("pool_ape"))
}

///|
test "index reuse source is a counts-bounded bit copy" {
  let source = reuse_selected_indices_i32_source(
    output_width=2048,
    maximum_tokens=4096,
    function_symbol="lunaflux_glm_hybrid_dsa_index_reuse_i32_v1",
  )
  assert_eq(source.numerical_contract(), BitExactI32Copy)
  assert_eq(source.output_width(), 2048)
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("const int query_count = counts[3];"))
  assert_true(text.contains("reused_indices[cell] = previous_indices[cell]"))
}
