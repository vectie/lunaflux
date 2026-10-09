///|
test "causal complete-slot source is exact bounded and model-neutral" {
  let rendered = causal_complete_slots_i32_source(
    compression_ratio=128,
    maximum_position_count=163840,
    maximum_rows=32,
    function_symbol="lunaflux_decoder_causal_complete_slots_i32_v1",
  )
  assert_eq(rendered.operation(), CausalCompleteSlotsI32)
  assert_eq(rendered.numerical_contract(), BitExactCausalCompleteSlotsI32V1)
  assert_eq(rendered.maximum_slots_per_row(), 1280)
  let text = @utf8.decode(rendered.source_bytes())
  assert_true(text.contains("const int32_t *query_positions"))
  assert_true(text.contains("const int32_t *cache_offsets"))
  assert_true(text.contains("int32_t *slot_counts"))
  assert_true(text.contains("int32_t *slot_indices"))
  assert_true(text.contains("declared_live_rows <= 32"))
  assert_true(text.contains("selected = (position + 1) / 128"))
  assert_true(text.contains("position < 163840"))
  assert_true(text.contains("offset > INT32_MAX - (selected - 1)"))
  assert_true(text.contains("slot < selected ? offset + slot : -1"))
  assert_eq(
    rendered.source_bytes(),
    causal_complete_slots_i32_source(
      compression_ratio=128,
      maximum_position_count=163840,
      maximum_rows=32,
      function_symbol="lunaflux_decoder_causal_complete_slots_i32_v1",
    ).source_bytes(),
  )
}

///|
test "causal complete-slot source rejects hostile geometry and symbols" {
  try
    causal_complete_slots_i32_source(
      compression_ratio=1,
      maximum_position_count=128,
      maximum_rows=1,
      function_symbol="compressed_index",
    )
  catch {
    InvalidCompressionRatio => ()
    _ => fail("unexpected compression-ratio rejection")
  } noraise {
    _ => fail("ratio-one source was admitted")
  }
  for position_limit in [0, 127] {
    try
      causal_complete_slots_i32_source(
        compression_ratio=128,
        maximum_position_count=position_limit,
        maximum_rows=1,
        function_symbol="compressed_index",
      )
    catch {
      InvalidMaximumPositionCount => ()
      _ => fail("unexpected maximum-position rejection")
    } noraise {
      _ => fail("invalid maximum position count was admitted")
    }
  }
  try
    causal_complete_slots_i32_source(
      compression_ratio=128,
      maximum_position_count=128,
      maximum_rows=1048577,
      function_symbol="compressed_index",
    )
  catch {
    InvalidMaximumRows => ()
    _ => fail("unexpected maximum-row rejection")
  } noraise {
    _ => fail("oversized row envelope was admitted")
  }
  try
    causal_complete_slots_i32_source(
      compression_ratio=128,
      maximum_position_count=128,
      maximum_rows=1,
      function_symbol="9bad",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("unexpected symbol rejection")
  } noraise {
    _ => fail("invalid symbol was admitted")
  }
}
