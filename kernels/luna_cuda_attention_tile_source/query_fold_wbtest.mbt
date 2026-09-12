///|
test "prefill scalarizes carried query state across compact and async schedules" {
  for candidate in [312, 313, 314, 316, 317] {
    let lowering = source_test_lowering(
      true,
      100000L,
      2,
      candidate_id=Some(candidate),
    )
    let source = @utf8.decode(
      emit_cuda_attention_tile_source(
        lowering,
        source_test_abi(),
        "static_query_fold",
        1048576,
      ).bytes(),
    )
    let (queries, _, _) = lowering.tile()
    let (_, _, threads, _) = lowering.launch()
    let offsets = @luna_attention_tile_schedule.scalarize_attention_query_fold(
      queries,
      threads / 32,
    )
    for offset in offsets {
      assert_true(
        source.contains("float fold_maximum_\{offset} = -CUDART_INF_F;"),
      )
      assert_true(source.contains("const int query = warp + \{offset};"))
      assert_true(
        source.contains("const int query_position = fold_position_\{offset};"),
      )
      assert_true(
        source.contains(
          "const int fold_position_\{offset} = warp + \{offset} < active_queries",
        ),
      )
      assert_true(
        source.contains(
          "fold_denominator_\{offset} * previous_scale + tile_sum",
        ),
      )
      assert_true(
        source.contains(
          "running_denominator[warp + \{offset}] = fold_denominator_\{offset};",
        ),
      )
    }
    assert_true(source.contains("if (query < active_queries)"))
    assert_false(source.contains("fold_maximum["))
    assert_false(source.contains("fold_denominator["))
    assert_false(source.contains("state_slot"))
    assert_false(source.contains("const int warp_count"))
    assert_false(source.contains("expf(lf_transformed_score("))
    assert_true(source.contains("expf(score_0 - next_maximum)"))
    assert_eq(
      source.contains("LfMatrixAccumulator lf_matrix_load_float("),
      !lowering.retains_output_fold(),
    )
  }
}

///|
test "noncompact prefill retains dynamic query traversal but omits unused float reload" {
  let lowering = source_test_lowering(true, 100000L, 2, candidate_id=Some(300))
  let source = @utf8.decode(
    emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "shared_query_fold",
      1048576,
    ).bytes(),
  )
  assert_true(source.contains("const int warp_count"))
  assert_false(source.contains("LfMatrixAccumulator lf_matrix_load_float("))
}
