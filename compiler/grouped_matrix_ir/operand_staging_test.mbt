///|
test "operand lifetime is independent of numerical law and tile ownership" {
  let direct = @grouped_matrix_ir.MatrixTilePlan::new(
    rows=17,
    columns=65,
    inner=136,
    law=ShapeAwareFloat32,
  )
  let buffered = @grouped_matrix_ir.MatrixTilePlan::new(
    rows=17,
    columns=65,
    inner=136,
    law=ShapeAwareFloat32,
    staging=Buffered(inner_tile=64),
  )
  assert_eq(buffered.law(), direct.law())
  assert_eq(buffered.tile_count(), direct.tile_count())
  assert_eq(direct.operand_slots(), 1)
  let regions = @grouped_matrix_ir.SegmentedMatrixPlan::new([buffered, buffered])
  assert_eq(regions.scratch_elements(), (2048, 8192, 1024))
  try @grouped_matrix_ir.SegmentedMatrixPlan::new([direct, buffered]) catch {
    _ => ()
  } noraise {
    _ => fail("incompatible shared scratch lifetimes")
  }
}

///|
test "expert fragment read-ahead does not change job or activation geometry" {
  let direct = @grouped_matrix_ir.ExpertGroupingPlan::new(
    maximum_rows=64,
    selected_experts=8,
  )
  let ahead = @grouped_matrix_ir.ExpertGroupingPlan::new(
    maximum_rows=64,
    selected_experts=8,
    fragment_slots=2,
  )
  assert_eq(direct.fragment_slots(), 1)
  assert_eq(ahead.fragment_slots(), 2)
  assert_eq(ahead.storage_bytes(), direct.storage_bytes())
  assert_eq(ahead.maximum_groups(), direct.maximum_groups())
  try
    @grouped_matrix_ir.ExpertGroupingPlan::new(
      maximum_rows=64,
      selected_experts=8,
      fragment_slots=3,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("unimplemented fragment depth accepted")
  }
}
