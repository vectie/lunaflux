///|
test "historical page batch borrows portable page ownership before vector reads" {
  let stage = cuda_query_owned_stage(
    source_test_abi(),
    true,
    true,
    split=true,
    key_rows=64,
  )
  let phases = stage.split("// page-batch-ownership-v1").to_array()
  assert_eq(phases.length(), 2)
  let body = phases[1].split("const int r=vector").to_array()
  assert_true(body[0].contains("staged_page=page_indices"))
  assert_true(body[0].contains("if (lane<4)"))
  assert_true(body[0].contains("atomicExch(control,0)"))
  assert_false(body[1].contains("page_indices["))
  assert_true(body[1].contains("__shfl_sync(0xffffffffU,staged_page,r/"))
  assert_true(body[1].contains("pending_values[index]"))
  assert_true(body[1].contains("const int bytes=kv_source_valid?16:0"))
  assert_false(phases[1].contains("__syncthreads"))
}

///|
test "unaligned or overfull page ownership retains the existing checked read map" {
  for rows in [0, 1, 513, 528, 2147483647] {
    assert_false(
      cuda_query_owned_stage(
        source_test_abi(),
        true,
        true,
        split=true,
        key_rows=rows,
      ).contains("page-batch-ownership-v1"),
    )
  }
}

///|
test "page batching belongs only to the split dense historical subdomain" {
  for (dense, split) in [(false, false), (true, false), (false, true)] {
    assert_false(
      cuda_query_owned_stage(
        source_test_abi(),
        dense,
        true,
        split~,
        key_rows=64,
      ).contains("page-batch-ownership-v1"),
    )
  }
  let stage = cuda_query_owned_stage(
    source_test_abi(),
    true,
    true,
    split=true,
    key_rows=512,
  )
  assert_true(stage.contains("if (lane<32)"))
  assert_true(stage.contains("kind==0 && kb%16==0"))
  assert_true(
    stage.contains("LF_KEY_VALUE_TILE<=min(row_position_begin,key_end)"),
  )
  assert_true(stage.contains("else stage_mixed_tile(kb,stage,kind)"))
}
