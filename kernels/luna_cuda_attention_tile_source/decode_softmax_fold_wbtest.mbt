///|
test "split decode rescales once per tile without altering transfer geometry" {
  for geometry in [(430, 34076L, 32), (440, 66844L, 32), (441, 33820L, 16)] {
    let (candidate, shared, slots) = geometry
    let lowering = source_test_decode_lowering(candidate_id=candidate)
    assert_eq(lowering.launch().3, shared)
    let source = @utf8.decode(
      emit_cuda_attention_tile_source(
        lowering,
        source_test_abi(),
        "tile_fold",
        1048576,
      ).bytes(),
    )
    assert_true(source.contains("#define LF_SCORE_SLOTS \{slots}"))
    assert_true(source.contains("float tile_score_0 = 0.0f"))
    assert_false(source.contains("float scores["))
    assert_true(source.contains("if (key < active_keys)"))
    assert_true(source.contains("if (lane == 0 && split < active_keys)"))
    assert_true(source.contains("float tile_old_scale = 1.0f"))
    assert_true(source.contains("expf(maximum - next_maximum)"))
    assert_true(source.contains("weight = expf(tile_score_0 - maximum)"))
    assert_true(source.contains("accumulator[item] *= tile_old_scale"))
    assert_false(source.contains("accumulator[item] * old_scale + new_scale"))
    assert_true(
      source.contains("value_tile[key * LF_HEAD_DIMENSION + component]"),
    )
    assert_true(
      source.contains("key_tile[key * LF_HEAD_DIMENSION + component]"),
    )
    assert_true(source.contains("query_position + 1"))
    assert_true(
      source.contains(
        "partial_numerators[warp * LF_HEAD_DIMENSION + component]",
      ),
    )
    assert_true(source.contains("#undef LF_SCORE_SLOTS"))
  }
}
