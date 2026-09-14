///|
test "finite operand rings publish before consumption without overwriting live slots" {
  for stages in [2, 3, 4] {
    for transfers in [4, 8, 16, 32, 64] {
      let plan : ProjectionFoldPipeline = {
        stages,
        fragments: Some(1),
        tile_width: 16,
        tile_count: transfers,
        transfer_width: 16,
        row_width: 16,
        output_tiles: 1,
        minimum_rows: 2,
      }
      let slots : Array[Int?] = Array::make(stages, None)
      for prime in 0..<(stages - 1) {
        slots[plan.slot(prime).unwrap()] = Some(prime)
      }
      for transfer in 0..<transfers {
        let current = plan.slot(transfer).unwrap()
        assert_eq(slots[current], Some(transfer))
        let future = transfer + stages - 1
        if future < transfers {
          let destination = plan.slot(future).unwrap()
          assert_false(destination == current)
          assert_eq(slots[destination], None)
          slots[destination] = Some(future)
        }
        slots[current] = None
      }
      assert_true(slots.all(fn(slot) { slot == None }))
      assert_eq(plan.slot(-1), None)
      assert_eq(plan.slot(transfers), None)
    }
  }
}
