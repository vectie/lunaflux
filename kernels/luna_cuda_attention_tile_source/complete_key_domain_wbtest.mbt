///|
test "blockwise full-domain specialization retains the ordered ragged fold" {
  for id in [450, 451, 452, 453, 454, 455, 456, 457] {
    let lowering = source_test_decode_lowering(
      candidate_id=id,
      alternative=true,
    )
    let text = @utf8.decode(
      emit_cuda_attention_tile_source(
        lowering,
        source_test_abi(),
        "complete_domain",
        1048576,
      ).bytes(),
    )
    assert_true(text.contains("key_group < active_keys"))
    assert_true(text.contains("key < active_keys"))
    assert_true(
      text.contains(
        "if (score_lane == 0 && key < active_keys) scores[key] = dot * scale;",
      ),
    )
    assert_false(text.contains("--use_fast_math"))
    guard lowering.physical_program() is BlockwiseKeyProgram(fold, _) else {
      fail("missing ordered fold")
    }
    assert_eq(
      text.contains("if (active_keys == LF_KEY_VALUE_TILE)"),
      fold.score_owners() > 1,
    )
    assert_eq(text.contains("key < LF_KEY_VALUE_TILE"), fold.score_owners() > 1)
    assert_eq(
      text.contains("key_group < LF_KEY_VALUE_TILE"),
      fold.score_owners() > 1,
    )
    assert_eq(
      text.contains("if (score_lane == 0) scores[key] = dot * scale;"),
      fold.score_owners() > 1,
    )
    assert_true(text.contains("numeric_law=" + fold.numeric_law()))
  }
}
