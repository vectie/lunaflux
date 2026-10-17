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
