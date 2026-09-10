///|
test "fold work cardinality follows independent row and column products" {
  let plan : ProjectionFoldPipeline = {
    tile_width: 16,
    tile_count: 64,
    transfer_width: 32,
    row_width: 16,
    output_tiles: 4,
    minimum_rows: 2,
  }
  assert_eq(
    plan.workgroup_count(
      maximum_rows=16,
      output_columns=3072,
      matrix_columns=16,
    ),
    Some(48),
  )
  assert_eq(
    plan.workgroup_count(
      maximum_rows=17,
      output_columns=3072,
      matrix_columns=16,
    ),
    Some(96),
  )
  assert_eq(
    plan.workgroup_count(
      maximum_rows=16,
      output_columns=3073,
      matrix_columns=16,
    ),
    Some(49),
  )
  assert_eq(
    plan.workgroup_count(maximum_rows=8, output_columns=3072, matrix_columns=16),
    Some(48),
  )
  let one_column = { ..plan, output_tiles: 1 }
  assert_eq(
    one_column.workgroup_count(
      maximum_rows=8,
      output_columns=3072,
      matrix_columns=16,
    ),
    Some(192),
  )
  let two_rows = { ..plan, row_width: 32 }
  assert_eq(
    two_rows.workgroup_count(
      maximum_rows=32,
      output_columns=3072,
      matrix_columns=16,
    ),
    Some(48),
  )
  for
    inputs in [
      (0, 3072, 16),
      (16, 0, 16),
      (16, 3072, 0),
      (2147483647, 2147483647, 1),
    ] {
    let (maximum_rows, output_columns, matrix_columns) = inputs
    assert_eq(
      plan.workgroup_count(maximum_rows~, output_columns~, matrix_columns~),
      None,
    )
  }
}
