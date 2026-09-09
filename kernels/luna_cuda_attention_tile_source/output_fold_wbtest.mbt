///|
test "resident output fold crosses KV tiles without shared accumulator round trips" {
  let lowering = source_test_lowering(false, 100000L, 2, candidate_id=Some(312))
  assert_true(lowering.retains_output_fold())
  let emitted = emit_cuda_prefill_attention_tile_source(
    lowering,
    source_test_abi(),
    "resident_fold_test",
    1048576,
  )
  let source = @utf8.decode(emitted.bytes())
  assert_true(
    source.contains("OutputFoldFragment output_fold[LF_QUERY_TILE / 16];"),
  )
  assert_true(source.contains("output_fold_rows[slice][item]"))
  assert_false(
    source.contains("accumulator[linear] *= previous_scales[query];"),
  )
  assert_false(source.contains("load_matrix_sync(output_fragment,"))
  assert_true(source.contains("slice * 16 + lf_matrix_row(item, lane)"))
  assert_true(
    source.contains("query < active_queries ? previous_scales[query] : 0.0f"),
  )
  let wide = source_test_lowering(false, 100000L, 2, candidate_id=Some(314))
  assert_false(wide.retains_output_fold())
}
