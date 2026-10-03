///|
test "independent operand ring reaches selected blockwise source with early refill" {
  let lowering = source_test_decode_lowering(candidate_id=460, alternative=true)
  guard lowering.physical_program() is BlockwiseKeyProgram(_, effects) else {
    fail("missing blockwise program")
  }
  assert_true(effects.independently_refills())
  guard source_test_decode_lowering(candidate_id=454, alternative=true).physical_program()
    is BlockwiseKeyProgram(incumbent_fold, incumbent_effects) else {
    fail("missing paired incumbent")
  }
  assert_false(incumbent_effects.independently_refills())
  guard lowering.physical_program() is BlockwiseKeyProgram(new_fold, _) else {
    fail("missing independent fold")
  }
  assert_eq(new_fold.numeric_law(), incumbent_fold.numeric_law())
  let source = @utf8.decode(
    emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "independent",
      1048576,
    ).bytes(),
  )
  assert_true(
    source.contains("stage_key(key_begin, 0); stage_value(key_begin, 0);"),
  )
  assert_true(
    source.contains(
      "stage_key(key_begin + LF_KEY_VALUE_TILE, 1); stage_value(key_begin + LF_KEY_VALUE_TILE, 1);",
    ),
  )
  assert_true(source.contains("cp.async.wait_group 3;"))
  assert_true(
    source.contains(
      "active_keys = max(0, min(LF_KEY_VALUE_TILE, key_limit - key_base))",
    ),
  )
  assert_eq(
    source.split("page_table_page_indices[epoch_slot]").to_array().length(),
    2,
  )
  assert_false(source.contains("unsigned long long retained_address_"))
  let keys = source
    .split("stage_key(key_base + 2 * LF_KEY_VALUE_TILE, slot);")
    .to_array()
  assert_eq(keys.length(), 2)
  assert_true(keys[0].contains("for (int key_group"))
  assert_false(keys[0].contains("float tile_maximum"))
  assert_true(keys[1].contains("float tile_maximum"))
  let values = source
    .split("stage_value(key_base + 2 * LF_KEY_VALUE_TILE, slot);")
    .to_array()
  assert_eq(values.length(), 2)
  assert_true(values[0].contains("float tile_numerator"))
  assert_false(values[0].contains("const float next_maximum"))
  assert_true(values[1].contains("const float next_maximum"))
  assert_true(source.contains("if (!stage_valid[2 + slot]) { asm volatile"))
}

///|
test "independent decode copy obeys retained backend storage map" {
  let lowering = source_test_decode_lowering(candidate_id=460, alternative=true)
  guard lowering.physical_program() is BlockwiseKeyProgram(_, effects) else {
    fail("missing effects")
  }
  let source = emit_decode_async_stage(
    source_test_abi(),
    effects,
    true,
    lowering.decode_copy_retention().unwrap(),
    vector_elements=8,
    tile_rows=32,
    start_alignment=32,
    storage_index="lf_operand_index(key, component, LF_HEAD_DIMENSION)",
  )
  assert_true(
    source.contains(
      "operand_tile + lf_operand_index(key, component, LF_HEAD_DIMENSION)",
    ),
  )
  assert_false(source.contains("operand_tile + linear"))
}

///|
test "independent decode validates malformed future pages and zero fills tails" {
  let lowering = source_test_decode_lowering(candidate_id=460, alternative=true)
  guard lowering.physical_program() is BlockwiseKeyProgram(_, effects) else {
    fail("missing effects")
  }
  let source = emit_decode_async_stage(
    source_test_abi(),
    effects,
    true,
    lowering.decode_copy_retention().unwrap(),
    vector_elements=8,
    tile_rows=32,
    start_alignment=32,
  )
  assert_true(source.contains("epoch_slot < page_end"))
  assert_true(source.contains("key_limit - key_base"))
  assert_true(
    source.contains("epoch_page >= 0 && (unsigned long long)epoch_page <"),
  )
  assert_true(source.contains("lane * 16 < active_keys && !page_valid"))
  assert_true(source.contains("atomicExch(stage_valid + slot, 0)"))
  assert_true(source.contains("atomicExch(stage_valid + 2 + slot, 0)"))
  assert_eq(
    source.split("\"r\"(row_valid ? 16 : 0)").to_array().length(),
    2 * lowering.decode_copy_retention().unwrap().dimensions().2 + 1,
  )
  assert_eq(source.split("cp.async.commit_group;").to_array().length(), 3)
  assert_eq(source.split("epoch_page <").to_array().length(), 2)
  assert_false(source.contains("const int position = key_base + key"))
  let value = source.split("auto stage_value").to_array()[1]
  assert_false(value.contains("epoch_page <"))
  assert_false(value.contains("epoch_page *"))
  let fallback = emit_decode_async_stage(
    source_test_abi(),
    effects,
    false,
    lowering.decode_copy_retention().unwrap(),
    vector_elements=8,
    tile_rows=32,
    start_alignment=32,
  )
  assert_false(fallback.contains("lookup_epoch_begin"))
  assert_true(fallback.contains("page_slot < page_end"))
  assert_true(fallback.contains("page < 0 || (unsigned long long)page >="))
}

///|
test "retained page origin lowering keeps wide caches wide" {
  let lowering = source_test_decode_lowering(candidate_id=460, alternative=true)
  guard lowering.physical_program() is BlockwiseKeyProgram(_, effects) else {
    fail("missing effects")
  }
  let abi = CudaPagedAttentionTileAbi::new(
    query_heads=16,
    key_value_heads=4,
    head_dimension=128,
    tokens_per_page=16,
    input_row_width=3072,
    page_stride_values=8192L,
    physical_page_capacity=524288L,
    maximum_query_rows=32,
    maximum_query_tokens=2048,
    maximum_page_entries=16384,
  )
  let source = emit_decode_async_stage(
    abi,
    effects,
    true,
    lowering.decode_copy_retention().unwrap(),
    vector_elements=8,
    tile_rows=32,
    start_alignment=32,
  )
  assert_true(source.contains("unsigned long long future_origin_0"))
  assert_true(source.contains("epoch_page < 524288ULL"))
  assert_true(
    source.contains("const unsigned long long page_origin = __shfl_sync"),
  )
}
