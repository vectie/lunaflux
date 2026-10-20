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
  try
    execution_policy({
      "sparse_projection_row_reuse": {
        "minimum_rows": 31.5,
        "minimum_inner": 1536,
        "wide_inner": 8192,
        "wide_tile": 64,
      },
    })
  catch {
    _ => ()
  } noraise {
    _ => fail("fractional row envelope accepted")
  }
}

///|
test "fragment pipeline is explicit startup shape selection" {
  let policy = execution_policy({
    "sparse_projection_row_reuse": {
      "minimum_rows": 32,
      "minimum_inner": 1536,
      "wide_inner": 8192,
      "wide_tile": 64,
      "fragment_inner": 8192,
      "fragment_columns": 4096,
    },
  })
  let old = @matrix.MatrixTilePlan::new(
    rows=64,
    columns=4096,
    inner=8192,
    law=ShapeAwareFloat32,
  )
  let selected = policy.sparse_row_reuse.unwrap().apply(old)
  assert_true(selected.staging() is FragmentPipeline(row_tile=32))
  assert_true(policy.description().contains("fragment_inner"))
}
