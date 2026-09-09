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
  }
}
