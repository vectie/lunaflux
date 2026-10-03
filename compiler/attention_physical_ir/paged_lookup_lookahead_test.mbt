///|
test "future page lookup epochs cover each page without shape magic numbers" {
  for
    (rows, page_rows, owners) in [
      (32, 8, 32),
      (64, 16, 32),
      (32, 16, 32),
      (64, 8, 16),
    ] {
    let plan = @attention_physical_ir.PagedLookupLookahead::new(
      tile_rows=rows,
      page_rows~,
      start_alignment=rows,
      owners~,
    ).unwrap()
    assert_eq(plan.tiles_per_epoch(), owners / (rows / page_rows))
    assert_eq(plan.rows_per_epoch(), plan.tiles_per_epoch() * rows)
    let seen = Array::make(owners, false)
    for tile in 0..<plan.tiles_per_epoch() {
      for page in 0..<(rows / page_rows) {
        let owner = plan.page_owner(tile, page).unwrap()
        assert_false(seen[owner])
        seen[owner] = true
      }
    }
    assert_true(seen.all(value => value))
    assert_eq(plan.page_owner(-1, 0), None)
    assert_eq(plan.page_owner(plan.tiles_per_epoch(), 0), None)
  }
  for
    (rows, page_rows, alignment, owners) in [
      (32, 8, 4, 32),
      (31, 8, 8, 32),
      (64, 8, 8, 4),
      (0, 8, 8, 32),
    ] {
    assert_eq(
      @attention_physical_ir.PagedLookupLookahead::new(
        tile_rows=rows,
        page_rows~,
        start_alignment=alignment,
        owners~,
      ),
      None,
    )
  }
  assert_eq(
    @attention_physical_ir.PagedLookupLookahead::new(
      tile_rows=2147483647,
      page_rows=2147483647,
      start_alignment=2147483647,
      owners=32,
    ),
    None,
  )
  let partial = @attention_physical_ir.PagedLookupLookahead::new(
    tile_rows=32,
    page_rows=8,
    start_alignment=32,
    owners=30,
  ).unwrap()
  assert_eq(partial.tiles_per_epoch(), 7)
  assert_eq(partial.page_owner(6, 3), Some(27))
  assert_eq(partial.page_owner(6, 4), None)
}

///|
test "future page epochs survive ring rollover and incomplete final pages" {
  for (tile_rows, page_rows) in [(32, 8), (32, 16), (64, 8), (64, 16)] {
    let plan = @attention_physical_ir.PagedLookupLookahead::new(
      tile_rows~,
      page_rows~,
      start_alignment=page_rows,
      owners=32,
    ).unwrap()
    let pages_per_tile = tile_rows / page_rows
    for begin in [0, page_rows, 3 * tile_rows] {
      for
        rows in [
          0,
          1,
          tile_rows - 1,
          tile_rows,
          tile_rows + 1,
          plan.rows_per_epoch() - 1,
          plan.rows_per_epoch(),
          plan.rows_per_epoch() + 1,
          2 * plan.rows_per_epoch() + 1,
        ] {
        let limit = begin + rows
        let mut lookup_begin = -1
        let lookup_pages = Array::make(32, -1)
        let key_pages = [
          Array::make(pages_per_tile, -1),
          Array::make(pages_per_tile, -1),
        ]
        let value_pages = [
          Array::make(pages_per_tile, -1),
          Array::make(pages_per_tile, -1),
        ]
        let issue_key = (base : Int, slot : Int) => {
          if base < limit &&
            (lookup_begin < 0 || base >= lookup_begin + plan.rows_per_epoch()) {
            lookup_begin = base
            for owner in 0..<32 {
              lookup_pages[owner] = if base + owner * page_rows < limit {
                // Nonidentity page IDs make a stale or wrong owner observable.
                (base / page_rows + owner) * 17 + 5
              } else {
                -1
              }
            }
          }
          for page in 0..<pages_per_tile {
            let relative_tile = (base - lookup_begin) / tile_rows
            key_pages[slot][page] = match plan.page_owner(relative_tile, page) {
              Some(owner) => lookup_pages[owner]
              None => -1
            }
          }
        }
        let issue_value = (slot : Int) => {
          for page in 0..<pages_per_tile {
            value_pages[slot][page] = key_pages[slot][page]
          }
        }
        issue_key(begin, 0)
        issue_value(0)
        issue_key(begin + tile_rows, 1)
        issue_value(1)
        let tiles = (rows + tile_rows - 1) / tile_rows
        for tile in 0..<tiles {
          let base = begin + tile * tile_rows
          let slot = tile % 2
          for page in 0..<pages_per_tile {
            if base + page * page_rows < limit {
              assert_eq(
                key_pages[slot][page],
                (base / page_rows + page) * 17 + 5,
              )
            }
          }
          // K can refresh the epoch and overwrite its retired slot while V
          // still reads the earlier transfer. The V copy owns its snapshot.
          issue_key(base + 2 * tile_rows, slot)
          for page in 0..<pages_per_tile {
            if base + page * page_rows < limit {
              assert_eq(
                value_pages[slot][page],
                (base / page_rows + page) * 17 + 5,
              )
            }
          }
          issue_value(slot)
        }
      }
    }
  }
}
