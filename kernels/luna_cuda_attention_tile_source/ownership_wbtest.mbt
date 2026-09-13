///|
test "query-owned source carries QK softmax PV across register ownership" {
  for id in [318, 319, 320, 321] {
    let lowering = source_test_lowering(true, 100000L, 2, candidate_id=Some(id))
    assert_true(lowering.query_owned_fragments())
    let source = emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "query_owned",
      1048576,
    )
    let text = @utf8.decode(source.bytes())
    assert_true(text.contains("LfMatrixAccumulator score["))
    assert_true(text.contains("probability.x[pair]=lo|(hi<<16)"))
    assert_true(
      text.contains("lf_matrix_mma(out[column],probability,value,true)"),
    )
    assert_false(text.contains("float *scores"))
    assert_true(text.contains("lf_query_score_fold<false>"))
    assert_true(text.contains("lf_query_score_fold<true>"))
    assert_false(text.contains("__nv_bfloat16 *weights"))
    assert_eq(
      source,
      emit_cuda_attention_tile_source(
        lowering,
        source_test_abi(),
        "query_owned",
        1048576,
      ),
    )
  }
}

///|
test "dense current and paged history share synchronous and async address lowering" {
  for id in [313, 316, 318, 319, 320, 321] {
    let lowering = source_test_lowering(
      true,
      100000L,
      2,
      candidate_id=Some(id),
      dense_current=true,
    )
    let source = emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "view",
      1048576,
    )
    let text = @utf8.decode(source.bytes())
    assert_true(text.contains("current_delta"))
    assert_true(text.contains("kv_key_source"))
    assert_true(text.contains("kv_page"))
    assert_true(text.contains("==key_position") || text.contains("==position"))
    if id == 316 || id == 320 || id == 321 {
      assert_true(text.contains("cp.async.cg.shared.global"))
    }
    if id == 320 || id == 321 {
      assert_true(text.contains("cp.async.wait_group 0"))
      assert_true(text.contains("slot^1"))
      assert_true(text.contains("current_k"))
    }
  }
}

///|
test "query-owned partition map exports merge states without shared scores" {
  let lowering = source_test_lowering(
    false,
    100000L,
    2,
    candidate_id=Some(318),
    partitioned=true,
  )
  assert_true(lowering.key_value_partitions() > 1)
  let source = emit_cuda_partitioned_attention_tile_source(
    lowering,
    source_test_abi(),
    "partial",
    "merge",
    134217728L,
    1048576,
  )
  let text = @utf8.decode(source.bytes())
  assert_true(text.contains("workspace[state]=maximum"))
  assert_true(text.contains("workspace[2ULL*"))
  assert_true(text.contains("void merge("))
  assert_false(text.contains("float *scores"))
}
