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
test "offline export of the partitioned blockwise physical frontier" {
  if @env.get_env_var("LUNA_EXPORT_BLOCKWISE_PARTITIONS") != Some("1") {
    return
  }
  let abi = CudaPagedAttentionTileAbi::new(
    query_heads=16,
    key_value_heads=8,
    head_dimension=128,
    tokens_per_page=8,
    input_row_width=4096,
    page_stride_values=8192L,
    physical_page_capacity=16384L,
    maximum_query_rows=32,
    maximum_query_tokens=2048,
    maximum_page_entries=16384,
  )
  for id in [441, 452, 453, 454, 455, 456, 457] {
    let lowering = source_test_decode_lowering(
      candidate_id=id,
      alternative=id >= 450,
      key_value_heads=8,
      tokens_per_page=8,
    )
    let source = emit_cuda_attention_tile_source(
      lowering, abi, "owned_decode", 1048576,
    )
    let law = match lowering.physical_program() {
      BlockwiseKeyProgram(fold, _) => fold.numeric_law()
      _ => "ordered-online-f32-v1"
    }
    let (_, heads, threads, shared) = lowering.launch()
    println("LUNA_PARTIAL_BEGIN c\{id}-p1")
    println(
      "grid=32,\{heads},1\nblock=\{threads},1,1\nshared_memory_bytes=\{shared}\nquery_tile_rows=1\nquery_metadata_version=0\nfunction_symbol=\{source.symbol()}\nmax_registers=128\ntarget=sm_121\nnumeric_law=\{law}",
    )
    println("LUNA_PARTIAL_SOURCE")
    println(@utf8.decode(source.bytes()))
    println("LUNA_PARTIAL_END")
  }
  for id in [450, 451, 452, 453, 454, 455, 456, 457] {
    for target in [16, 32, 64] {
      let lowering = source_test_decode_lowering(
        candidate_id=id,
        alternative=true,
        batch_rows=1,
        parallel_workgroup_target=target,
        key_value_heads=8,
        tokens_per_page=8,
      )
      let source = emit_cuda_partitioned_attention_tile_source(
        lowering, abi, "blockwise_partial", "blockwise_merge", 16777216L, 1048576,
      )
      let law = match lowering.physical_program() {
        BlockwiseKeyProgram(fold, _) => fold.numeric_law()
        _ => fail("missing exported blockwise program")
      }
      let (_, heads, threads, shared) = lowering.launch()
      println("LUNA_PARTIAL_BEGIN c\{id}-p\{lowering.key_value_partitions()}")
      println(
        "grid=32,\{heads},\{lowering.key_value_partitions()}\nmerge_grid=32,16,1\nblock=\{threads},1,1\nshared_memory_bytes=\{shared}\nworkspace_bytes=\{source.workspace().total_bytes()}\nquery_tile_rows=1\nquery_metadata_version=0\nfunction_symbol=blockwise_partial\nmerge_function_symbol=blockwise_merge\nmax_registers=128\ntarget=sm_121\nnumeric_law=\{law}",
      )
      println("LUNA_PARTIAL_SOURCE")
      println(@utf8.decode(source.bytes()))
      println("LUNA_PARTIAL_END")
    }
  }
}

///|
test "blockwise partial fold binds partition ownership and an explicit merge ABI" {
  for id in [450, 452] {
    let lowering = source_test_decode_lowering(
      candidate_id=id,
      alternative=true,
      batch_rows=1,
      parallel_workgroup_target=96,
    )
    assert_eq(lowering.key_value_partitions(), 8)
    guard lowering.physical_program() is BlockwiseKeyProgram(fold, _) else {
      fail("partitioned blockwise program missing")
    }
    assert_eq(fold.dimensions(), (128, 32))
    let source = emit_cuda_partitioned_attention_tile_source(
      lowering,
      source_test_abi(),
      "blockwise_partial",
      "blockwise_merge",
      16777216L,
      1048576,
    )
    let text = @utf8.decode(source.bytes())
    assert_true(text.contains("void blockwise_partial("))
    assert_false(text.contains("void blockwise_partial_blockwise_f32_v1("))
    assert_true(text.contains("void blockwise_merge("))
    assert_true(text.contains("float *workspace"))
    assert_true(text.contains("workspace[state] = CUDART_NAN_F"))
    assert_true(text.contains("workspace[state] = maximum"))
    assert_true(text.contains("const int key_begin = first *"))
    assert_true(
      text.contains(
        "state * LF_HEAD_DIMENSION + component] = accumulator[item]",
      ),
    )
    assert_eq(source.workspace().total_bytes(), 32L * 16L * 8L * 130L * 4L)
    try
      emit_cuda_attention_tile_source(
        lowering,
        source_test_abi(),
        "unsafe_single",
        1048576,
      )
    catch {
      InvalidCudaAttentionTileSource(UnsupportedLowering) => ()
      _ => fail("wrong single-launch rejection")
    } noraise {
      _ => fail("partial workspace escaped through a single-output ABI")
    }
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
