///|
test "complete affine current interval removes paged addressing only from its proven subdomain" {
  let abi = source_test_abi()
  let source = cuda_query_owned_stage(
    abi,
    true,
    true,
    split=true,
    affine_positions=true,
  )
  assert_true(source.contains("kb>=row_position_begin"))
  assert_true(
    source.contains("LF_KEY_VALUE_TILE<=row_position_begin+(end-begin)"),
  )
  assert_true(source.contains("LF_KEY_VALUE_TILE<=key_end"))
  let current = source.split("auto stage_current_tile").to_array()[1]
    .split("auto stage_complete_history_tile")
    .to_array()[0]
  assert_false(current.contains("page_indices"))
  assert_false(current.contains("__shfl_sync"))
  assert_true(current.contains("pending_values[index]=kv_value_source"))
  assert_true(current.contains("const bool kv_source_valid=true"))
  // The retained V address is consumed once by the ordinary V publication.
  assert_true(source.contains("kv_value_source=pending_values[index]"))
  assert_true(source.contains("stage_history_tile(kb,stage,kind)"))
  assert_true(source.contains("stage_mixed_tile(kb,stage,kind)"))
  for
    (dense, split, affine) in [
      (false, true, true),
      (true, false, true),
      (true, true, false),
    ] {
    assert_false(
      cuda_query_owned_stage(abi, dense, true, split~, affine_positions=affine).contains(
        "stage_current_tile",
      ),
    )
  }
}
