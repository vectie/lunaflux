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
