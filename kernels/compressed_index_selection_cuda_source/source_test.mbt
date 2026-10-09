///|
test "weighted compressed index preserves numeric publications without double scaling" {
  let plan = @precision.CompressedIndexSelectionPrecision::new(
    head_count=2,
    head_width=128,
    compression_ratio=4,
    maximum_rows=3,
    maximum_positions=16,
    top_k=2,
    maximum_workspace_bytes=60L,
  )
  let source = @compressed_index_selection_cuda_source.compressed_index_selection_source(
    plan,
    symbol_prefix="index_probe",
  )
  let text = @utf8.decode(source.source_bytes())
  assert_eq(source.symbol(0), "index_probe_scores")
  assert_eq(source.symbol(1), "index_probe_select")
  assert_eq(source.launch(0), (3, 1, 256))
  assert_eq(source.launch(1), (3, 1, 256))
  assert_true(text.contains("__float2bfloat16_rn(dot)"))
  assert_true(text.contains("__float2bfloat16_rn(__fmul_rn("))
  assert_true(text.contains("__float2bfloat16_rn(score)"))
  assert_false(text.contains("rsqrt"))
  assert_false(text.contains("sqrtf"))
  assert_false(text.contains("for (int prior"))
  assert_true(text.contains("candidate >= (position + 1) / 4"))
  assert_true(text.contains("offset + best_indices[0]"))
  assert_true(text.contains("selected_counts[row] = 0"))
}
