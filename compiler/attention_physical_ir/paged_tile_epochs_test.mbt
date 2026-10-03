///|
test "aligned page epochs preserve fragmented page-table identity" {
  for page_rows in [1, 2, 4, 8, 16, 32, 64] {
    for tile_rows in [32, 64, 128] {
      if tile_rows % page_rows != 0 || tile_rows / page_rows > 32 {
        continue
      }
      let plan = @attention_physical_ir.PagedTileEpochs::new(
        tile_rows~,
        page_rows~,
        start_alignment=tile_rows,
        owners=32,
      ).unwrap()
      assert_eq(plan.page_count(), tile_rows / page_rows)
      for start in [0, tile_rows, tile_rows * 7] {
        for active in [1, tile_rows - 1, tile_rows] {
          for row in 0..<active {
            let page = plan.page_owner(row).unwrap()
            // A physical page need not be adjacent to its logical neighbour.
            let physical = (start / page_rows + page) * 17 + 3
            let original = (start + row) / page_rows * 17 + 3
            assert_eq(physical, original)
            assert_eq((start + row) % page_rows, row % page_rows)
          }
        }
      }
      assert_true(plan.page_owner(-1) is None)
      assert_true(plan.page_owner(tile_rows) is None)
    }
  }
}

///|
test "unaligned and oversized epoch ownership remains on the general path" {
  for
    (tile_rows, page_rows, start_alignment, owners) in [
      (32, 12, 32, 32),
      (32, 16, 8, 32),
      (64, 1, 64, 32),
      (0, 16, 32, 32),
      (32, 0, 32, 32),
      (32, 16, 0, 32),
      (32, 16, 32, 0),
    ] {
    assert_true(
      @attention_physical_ir.PagedTileEpochs::new(
        tile_rows~,
        page_rows~,
        start_alignment~,
        owners~,
      )
      is None,
    )
  }
}
