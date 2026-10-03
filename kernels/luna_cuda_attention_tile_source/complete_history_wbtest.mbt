///|
test "complete historical page lookup sharing preserves the owner page" {
  for dimension in [16, 32, 64, 128] {
    for page_tokens in [1, 2, 4, 8, 16, 32] {
      for threads in [64, 128] {
        let width = (page_tokens * (dimension / 8)).min(32)
        for tile in [32, 64, 128] {
          for begin in [0, page_tokens, page_tokens * 7] {
            for vector in 0..<(tile * dimension / 8) {
              let lane = vector % threads % 32
              let owner_vector = vector - lane + lane / width * width
              assert_eq(
                (begin + vector * 8 / dimension) / page_tokens,
                (begin + owner_vector * 8 / dimension) / page_tokens,
              )
            }
          }
        }
      }
    }
  }
}

///|
test "complete history specialization retains page validation and checked fallback" {
  let text = cuda_query_owned_stage(
    source_test_abi(),
    true,
    true,
    split=true,
    affine_positions=true,
  )
  let body = text.split("auto stage_complete_history_tile").to_array()[1]
    .split("auto stage_query_tile")
    .to_array()[0]
  assert_true(body.contains("atomicExch(control,0)"))
  assert_true(
    body.contains(
      "pending_values[index]=kv_source_valid?kv_value_source:nullptr",
    ),
  )
  assert_false(body.contains("current_delta"))
  assert_true(text.contains("<=min(row_position_begin,key_end)"))
  assert_true(text.contains("stage_history_tile(kb,stage,kind)"))
  assert_true(text.contains("stage_mixed_tile(kb,stage,kind)"))
}

///|
test "historical owner width saturates before large page multiplication" {
  for dimension in [64, 128] {
    let abi = {
      ..source_test_abi(),
      head_dimension: dimension,
      tokens_per_page: 1073741824,
    }
    let body = cuda_query_complete_history_stage(abi, "", "", "")
    assert_true(body.contains("lane%32==0"))
    assert_true(body.contains("(lane/32)*32"))
    assert_false(body.contains("lane%0"))
  }
}
