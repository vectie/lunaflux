///|
test "sparse row ownership is explicit startup configuration" {
  assert_true(execution_policy({}).sparse_row_reuse is None)
  let policy = execution_policy({
    "sparse_projection_row_reuse": {
      "minimum_rows": 32,
      "minimum_inner": 1536,
      "wide_inner": 8192,
      "wide_tile": 64,
    },
  })
  assert_true(policy.sparse_row_reuse is Some(_))
  assert_eq(policy.recurrent_tile_rows, 16)
  assert_true(policy.description().contains("sparse_projection_row_reuse=Some"))
  assert_true(
    (try? execution_policy({
      "sparse_projection_row_reuse": {
        "minimum_rows": 31.5,
        "minimum_inner": 1536,
        "wide_inner": 8192,
        "wide_tile": 64,
      },
    }))
    is Err(_),
  )
}
