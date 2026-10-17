///|
test "matrix tiles cover tails without changing numerical authorization" {
  let p = @grouped_matrix_ir.MatrixTilePlan::new(
    rows=17,
    columns=65,
    inner=33,
    law=MatrixFloat32,
  )
  assert_eq(p.geometry(), (17, 65, 33))
  assert_eq(p.tile(), (16, 64, 32))
  assert_eq(p.tile_count(), 4)
  try
    @grouped_matrix_ir.MatrixTilePlan::new(
      rows=1,
      columns=1,
      inner=1,
      law=ScalarFloat32,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("unauthorized reassociation")
  }
}

///|
test "group storage bounds worst case distinct experts and padding" {
  let p = @grouped_matrix_ir.ExpertGroupingPlan::new(
    maximum_rows=17,
    selected_experts=8,
  )
  assert_eq(p.maximum_groups(), 136)
  assert_eq(p.rows_per_group(), 16)
  assert_eq(p.storage_bytes(), 9252L)
  assert_eq(p.column_tile(), 32)
  try
    @grouped_matrix_ir.ExpertGroupingPlan::new(
      maximum_rows=2147483647,
      selected_experts=8,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("overflow accepted")
  }
}

///|
test "column reuse leaves grouping storage and row ownership unchanged" {
  for columns in [32, 64, 128] {
    let p = @grouped_matrix_ir.ExpertGroupingPlan::new(
      maximum_rows=17,
      selected_experts=8,
      column_tile=columns,
    )
    assert_eq(p.column_tile(), columns)
    assert_eq(p.rows_per_group(), 16)
    assert_eq(p.storage_bytes(), 9252L)
  }
  for columns in [0, -1, 1025] {
    try
      @grouped_matrix_ir.ExpertGroupingPlan::new(
        maximum_rows=17,
        selected_experts=8,
        column_tile=columns,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("unbounded column ownership")
    }
  }
}
