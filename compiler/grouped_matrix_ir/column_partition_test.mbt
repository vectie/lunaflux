///|
test "independent column partitions retain numerical law and cover tails" {
  for columns in [16, 32, 64] {
    let plan = @grouped_matrix_ir.MatrixTilePlan::new(
      rows=17,
      columns=65,
      inner=4096,
      law=MatrixFloat32,
      tile_columns=columns,
    )
    assert_eq(plan.tile(), (16, columns, 32))
    assert_eq(plan.tile_count(), 2 * ((65 + columns - 1) / columns))
    let group = @grouped_matrix_ir.ExpertGroupingPlan::new(
      maximum_rows=64,
      selected_experts=1,
      matrix_columns=columns,
    )
    assert_eq(group.matrix_columns(), columns)
    assert_eq(group.storage_bytes(), 4356L)
  }
  try
    @grouped_matrix_ir.MatrixTilePlan::new(
      rows=16,
      columns=1024,
      inner=4096,
      law=MatrixFloat32,
      tile_columns=8,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("unsupported column ownership")
  }
  try
    @grouped_matrix_ir.ExpertGroupingPlan::new(
      maximum_rows=64,
      selected_experts=8,
      matrix_columns=16,
      transport=PackedPipeline,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("unconsumed physical option")
  }
}
