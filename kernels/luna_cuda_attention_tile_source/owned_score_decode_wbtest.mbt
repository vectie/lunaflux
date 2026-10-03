///|
test "owned score decode alternatives reach exact source and numeric contracts" {
  for id in [460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471] {
    let lowering = source_test_decode_lowering(
      candidate_id=id,
      alternative=true,
    )
    guard lowering.physical_program() is BlockwiseKeyProgram(fold, _) else {
      fail("missing owned blockwise program")
    }
    let source = emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "owned",
      1048576,
    )
    let text = @utf8.decode(source.bytes())
    assert_true(text.contains("numeric_law=\{fold.numeric_law()}"))
    assert_true(text.contains("#define LF_SCORE_OWNERS \{fold.score_owners()}"))
    assert_eq(text.contains("dot = fmaf("), fold.contracts_multiply_add())
    if fold.contracts_multiply_add() {
      assert_false(text.contains("dot += query_fragment[item] *"))
      assert_false(text.contains("tile_numerator[item] += probability *"))
      assert_eq(text.split("dot = fmaf(").length(), 3)
      assert_eq(text.split("tile_numerator[item] = fmaf(").length(), 3)
    }
    assert_false(text.contains("__float2bfloat16_rn(probability)"))
    assert_false(text.contains("__expf("))
    assert_false(text.contains("__exp2f("))
    try source_test_decode_lowering(candidate_id=id) catch {
      _ => ()
    } noraise {
      _ => fail("strict numerical contract admitted reassociation")
    }
  }
}
