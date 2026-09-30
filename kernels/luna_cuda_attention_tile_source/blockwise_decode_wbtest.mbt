///|
test "blockwise decode retains explicit F32 law and uniform cache rejection" {
  for id in [450, 451] {
    let lowering = source_test_decode_lowering(
      candidate_id=id,
      alternative=true,
    )
    guard lowering.physical_program() is BlockwiseKeyProgram(fold, effects) else {
      fail("missing blockwise physical program")
    }
    assert_eq(fold.numeric_law(), "blockwise-f32-probability-v1")
    assert_false(effects.is_pipelined())
    let source = emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "decode",
      1048576,
    )
    assert_eq(source.symbol(), "decode_blockwise_f32_v1")
    let text = @utf8.decode(source.bytes())
    assert_true(text.contains("float *scores"))
    assert_true(
      text.contains(
        "const float probability = expf(scores[key] - tile_maximum)",
      ),
    )
    assert_true(text.contains("min(LF_KEY_VALUE_TILE, key_limit - key_base)"))
    assert_true(text.contains("const int row = prefill_count + local_row"))
    assert_true(text.contains("page_end - page_start !="))
    assert_true(text.contains("atomicExch(tile_valid, 0)"))
    assert_true(text.contains("if (!*tile_valid) return;"))
    assert_false(text.contains("__float2bfloat16_rn(probability)"))
    assert_false(text.contains("old_scale + new_scale"))
    assert_eq(
      source,
      emit_cuda_attention_tile_source(
        lowering,
        source_test_abi(),
        "decode",
        1048576,
      ),
    )
    let (_, _, threads, shared) = lowering.launch()
    let (_, keys) = fold.dimensions()
    assert_eq(threads, 128)
    assert_eq(shared, (keys * 128 * 4 + 4 * keys * 4 + 16).to_int64())
  }
}

///|
test "blockwise decode pipelines operands with disjoint score ownership" {
  for id in [452, 453] {
    let lowering = source_test_decode_lowering(
      candidate_id=id,
      alternative=true,
    )
    guard lowering.physical_program() is BlockwiseKeyProgram(fold, effects) else {
      fail("missing blockwise physical program")
    }
    assert_true(effects.is_pipelined())
    let source = emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "decode",
      1048576,
    )
    let text = @utf8.decode(source.bytes())
    assert_true(text.contains("cp.async.ca.shared.global"))
    assert_true(text.contains("cp.async.wait_group 1;"))
    assert_true(text.contains("cp.async.wait_group 0;"))
    assert_true(
      text.contains("stage_tile(key_base + LF_KEY_VALUE_TILE, slot ^ 1)"),
    )
    assert_true(text.contains("if (!stage_valid[slot]) return;"))
    assert_true(text.contains("stage_arena + 4 * LF_KEY_VALUE_TILE"))
    assert_eq(source.symbol(), "decode_blockwise_f32_v1")
    let (_, keys) = fold.dimensions()
    assert_eq(
      lowering.launch().3,
      (keys * 128 * 8 + 4 * keys * 4 + 16).to_int64(),
    )
  }
}

///|
test "ordered decode does not expose the alternative numerical law" {
  try source_test_decode_lowering(candidate_id=450) catch {
    _ => ()
  } noraise {
    _ => fail("ordered policy admitted blockwise alternative")
  }
  let ordered = emit_cuda_attention_tile_source(
    source_test_decode_lowering(candidate_id=420),
    source_test_abi(),
    "decode",
    1048576,
  )
  assert_eq(ordered.symbol(), "decode")
  assert_false(@utf8.decode(ordered.bytes()).contains("numeric_law=blockwise"))
}

///|
test "blockwise decode rejects metadata ABI and oversized suffixed symbols" {
  let lowering = source_test_decode_lowering(candidate_id=450, alternative=true)
  let abi = source_test_abi()
  try
    emit_cuda_attention_tile_source(
      lowering,
      { ..abi, query_metadata: true, },
      "decode",
      1048576,
    )
  catch {
    InvalidCudaAttentionTileSource(UnsupportedLowering) => ()
    _ => fail("unexpected metadata rejection")
  } noraise {
    _ => fail("ten-pointer blockwise kernel accepted metadata ABI")
  }
  let symbol = "a".repeat(128)
  try emit_cuda_attention_tile_source(lowering, abi, symbol, 1048576) catch {
    InvalidCudaAttentionTileSource(InvalidSymbol) => ()
    _ => fail("unexpected symbol rejection")
  } noraise {
    _ => fail("suffixed CUDA symbol exceeds public symbol bound")
  }
}
