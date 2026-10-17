///|
test "padded rows share exact producer consumer addresses without owning holes" {
  for stride in [128, 144, 160] {
    let layout = @physical_tile_ir.OperandLayout::new(
      columns=128,
      vector=16,
      rotations=1,
      row_group=1,
      row_stride=stride,
    ).unwrap()
    let storage = @physical_tile_ir.OperandStorage::new(
      layout,
      [16, 64, 64],
      owners=32,
      stages=2,
    ).unwrap()
    assert_eq(storage.slot_elements(), 144 * stride)
    assert_eq(storage.allocation_elements(), 288 * stride)
    let seen = FixedArray::make(storage.slot_elements(), false)
    for region_index in 0..<3 {
      let region = storage.region(region_index).unwrap()
      let (_, _, rounds) = region.ownership().geometry()
      for owner in 0..<32 {
        for round in 0..<rounds {
          if region.copy_at(owner, round) is Some((row, col, offset)) {
            for byte in 0..<16 {
              assert_eq(region.offset(row, col + byte), Some(offset + byte))
              assert_false(seen[offset + byte])
              seen[offset + byte] = true
            }
          }
        }
      }
    }
    for i in 0..<seen.length() {
      assert_eq(seen[i], i % stride < 128)
    }
    assert_true(storage.has_concatenated_layout())
  }
  for stride in [0, 127, 129] {
    assert_true(
      @physical_tile_ir.OperandLayout::new(
        columns=128,
        vector=16,
        rotations=1,
        row_group=1,
        row_stride=stride,
      )
      is None,
    )
  }
  let huge = @physical_tile_ir.OperandLayout::new(
    columns=128,
    vector=16,
    rotations=1,
    row_group=1,
    row_stride=2147483632,
  ).unwrap()
  assert_true(
    @physical_tile_ir.OperandStorage::new(huge, [2], owners=32, stages=1)
    is None,
  )
  assert_true(huge.offset(2, 0) is None)
}
