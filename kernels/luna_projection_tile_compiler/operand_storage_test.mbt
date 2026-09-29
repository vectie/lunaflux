///|
test "physical operand storage is joined to logical transfer geometry" {
  let transfers = @luna_projection_tile_compiler.ProjectionOperandTransferPlan::new(
    input_rows=32,
    weight_rows=64,
    row_width=64,
    vector_width=8,
    owners=256,
  ).unwrap()
  let layout = @physical_tile_ir.OperandLayout::new(
    columns=64,
    vector=8,
    rotations=8,
    row_group=1,
  ).unwrap()
  let storage = transfers.storage(layout, stages=3).unwrap()
  assert_eq(
    storage.region(0).unwrap().ownership().geometry(),
    transfers.input_map().geometry(),
  )
  assert_eq(
    storage.region(1).unwrap().ownership().geometry(),
    transfers.weight_map().geometry(),
  )
  assert_eq(storage.region(1).unwrap().geometry().0, transfers.weight_offset())
  assert_eq(storage.slot_elements(), transfers.slot_elements())
  assert_eq(storage.allocation_elements(), 3 * transfers.slot_elements())
  for (columns, vector) in [(32, 8), (64, 4)] {
    let different = @physical_tile_ir.OperandLayout::new(
      columns~,
      vector~,
      rotations=1,
      row_group=1,
    ).unwrap()
    assert_true(transfers.storage(different, stages=2) is None)
  }
}
