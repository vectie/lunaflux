///|
test "prefill matrix producers and consumers share compact operand addressing" {
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
        "compact_operand_tiles",
        1048576,
      ).bytes(),
    )
    assert_true(source.contains("#define LF_OPERAND_TILE 8"))
    assert_true(
      source.contains(
        "query_tile + lf_operand_index(query, component, LF_HEAD_DIMENSION)",
      ),
    )
    assert_true(
      source.contains(
        "stage_keys + lf_operand_index(key, component, LF_HEAD_DIMENSION)",
      ),
    )
    assert_true(
      source.contains(
        "stage_values + lf_operand_index(key, component, LF_HEAD_DIMENSION)",
      ),
    )
    assert_true(
      source.contains(
        "weights[lf_operand_index(query, key, LF_KEY_VALUE_TILE)]",
      ),
    )
    assert_true(
      source.contains("reduction, warp * 16, LF_HEAD_DIMENSION, true)") ||
      source.contains("reduction, output_column, LF_HEAD_DIMENSION, true)"),
    )
    assert_false(source.contains("query_tile + first_value"))
    assert_false(source.contains("stage_keys + linear"))
    assert_false(source.contains("stage_values + linear"))
    assert_false(
      source.contains("weights + slice * 16 * LF_KEY_VALUE_TILE + reduction"),
    )
  }
}

///|
test "scalar PV keeps float probabilities logical and maps compact values" {
  for candidate in [290, 291, 300] {
    let lowering = source_test_lowering(
      false,
      100000L,
      1,
      candidate_id=Some(candidate),
    )
    let source = @utf8.decode(
      emit_cuda_attention_tile_source(
        lowering,
        source_test_abi(),
        "compact_scalar_pv",
        1048576,
      ).bytes(),
    )
    assert_true(source.contains("float *weights = reinterpret_cast<float *>"))
    assert_true(
      source.contains("weights[query * LF_KEY_VALUE_TILE + key] = weight;"),
    )
    assert_true(
      source.contains("partial += weights[query * LF_KEY_VALUE_TILE + key]"),
    )
    assert_true(
      source.contains("lf_operand_index(key, component, LF_HEAD_DIMENSION)]"),
    )
    assert_false(source.contains("weights[lf_operand_index"))
    assert_true(
      source.contains(
        "lf_matrix_load(query_tile, query_base, reduction, LF_HEAD_DIMENSION)",
      ),
    )
  }
}

///|
test "compact matrix layout does not alter grouped decode storage" {
  let lowering = source_test_decode_lowering()
  let source = @utf8.decode(
    emit_cuda_attention_tile_source(
      lowering,
      source_test_abi(),
      "unchanged_decode",
      1048576,
    ).bytes(),
  )
  assert_false(source.contains("lf_operand_index"))
  assert_false(source.contains("LF_OPERAND_TILE"))
  assert_true(
    source.contains("stage_keys[key * LF_HEAD_DIMENSION + component]"),
  )
}
