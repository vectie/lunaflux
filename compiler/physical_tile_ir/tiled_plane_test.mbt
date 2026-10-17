///|
test "contiguous tile permutation preserves every byte and row-local vector" {
  for tile_rows in [16, 64, 128] {
    for tile_width in [16, 128] {
      let layout = @physical_tile_ir.OperandLayout::new(
        columns=tile_width,
        vector=16,
        rotations=tile_width / 16,
        row_group=1,
      ).unwrap()
      let plane = @physical_tile_ir.TiledPlane::new(
        rows=tile_rows * 3,
        columns=tile_width * 5,
        tile_rows~,
        layout~,
      ).unwrap()
      let seen = FixedArray::make(plane.elements(), false)
      for row in 0..<(tile_rows * 3) {
        for column in 0..<(tile_width * 5) {
          let offset = plane.offset(row, column).unwrap()
          assert_false(seen[offset])
          seen[offset] = true
          if column % 16 != 0 {
            assert_eq(offset, plane.offset(row, column - 1).unwrap() + 1)
          }
        }
      }
      assert_true(seen.iter().all(x => x))
      assert_true(plane.offset(-1, 0) is None)
      assert_true(plane.offset(tile_rows * 3, 0) is None)
    }
  }
}

///|
test "tile plane rejects incomplete or padded domains and bounded address overflow" {
  let layout = @physical_tile_ir.OperandLayout::new(
    columns=128,
    vector=16,
    rotations=8,
    row_group=1,
  ).unwrap()
  for
    (rows, columns, tile_rows) in [
      (0, 128, 16),
      (16, 0, 16),
      (16, 128, 0),
      (17, 128, 16),
      (16, 129, 16),
      (16, 128, 4),
      (16777216, 128, 16),
    ] {
    assert_true(
      @physical_tile_ir.TiledPlane::new(rows~, columns~, tile_rows~, layout~)
      is None,
    )
  }
  let padded = @physical_tile_ir.OperandLayout::new(
    columns=128,
    vector=16,
    rotations=8,
    row_group=1,
    row_stride=144,
  ).unwrap()
  assert_true(
    @physical_tile_ir.TiledPlane::new(
      rows=16,
      columns=128,
      tile_rows=16,
      layout=padded,
    )
    is None,
  )
}
