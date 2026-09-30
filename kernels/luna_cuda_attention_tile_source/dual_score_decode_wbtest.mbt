///|
test "dual score ownership is consumed by asynchronous decode lowering" {
  for id in [454, 455] {
    let lowering = source_test_decode_lowering(
      candidate_id=id,
      alternative=true,
    )
    guard lowering.physical_program() is BlockwiseKeyProgram(fold, effects) else {
      fail("missing explicit dual score fold")
    }
    assert_eq(fold.score_owners(), 2)
    assert_eq(fold.numeric_law(), "dual-score-blockwise-f32-probability-v2")
    assert_true(effects.is_pipelined())
    let source = emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "decode",
      1048576,
    )
    assert_eq(source.symbol(), "decode_dual_score_blockwise_f32_v2")
    let text = @utf8.decode(source.bytes())
    assert_true(text.contains("#define LF_SCORE_OWNERS 2"))
    assert_true(text.contains("#define LF_SCORE_LANES 16"))
    assert_true(text.contains("#define LF_SCORE_QUERY_ITEMS 8"))
    assert_true(text.contains("key_group += LF_SCORE_OWNERS"))
    assert_true(text.contains("offset, LF_SCORE_LANES)"))
    assert_true(
      text.contains("key < active_keys && component < LF_HEAD_DIMENSION"),
    )
    assert_true(text.contains("cp.async.ca.shared.global"))
    assert_true(
      text.contains("const float next_maximum = fmaxf(maximum, tile_maximum)"),
    )
    assert_false(text.contains("__float2bfloat16_rn(probability)"))
    assert_eq(
      source,
      emit_cuda_attention_tile_source(
        lowering,
        source_test_abi(),
        "decode",
        1048576,
      ),
    )
  }
}

///|
test "decode merge publishes invariant statistics only once" {
  let lowering = source_test_decode_lowering(
    candidate_id=454,
    alternative=true,
    batch_rows=1,
    parallel_workgroup_target=96,
  )
  let plan = lowering.decode_partition_merge().unwrap()
  assert_eq(plan.dimensions(), (8, 128))
  let source = emit_cuda_partitioned_attention_tile_source(
    lowering,
    source_test_abi(),
    "partial",
    "merge",
    16777216L,
    1048576,
  )
  let text = @utf8.decode(source.bytes())
  assert_true(
    text.contains("partition_merge_law=ascending-partition-f32-statistics-v1"),
  )
  assert_true(text.contains("if (thread == 0)"))
  assert_true(
    text.contains("__shared__ float partition_scales[LF_MERGE_PARTITIONS + 1]"),
  )
  assert_true(text.contains("partition_active[split]"))
  assert_false(text.contains("int row = -1, token_base = 0, tile_prefix = 0"))
  assert_false(text.contains("merged_numerator"))
  assert_eq(source.workspace().total_bytes(), 32L * 16L * 8L * 130L * 4L)
}

///|
test "ordered numeric policy excludes dual score alternatives" {
  for id in [454, 455] {
    try source_test_decode_lowering(candidate_id=id) catch {
      _ => ()
    } noraise {
      _ => fail("ordered policy admitted dual score reassociation")
    }
  }
}
