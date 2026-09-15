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
      for execution in [SerialTransfer, OverlappedTransfer] {
        let effects = plan.effect_plan(execution)
        assert_eq(effects.geometry(), (stages, 16, transfers * 16))
        assert_eq(effects.prime_count(), stages - 1)
        assert_eq(effects.future_distance(), (stages - 1) * 16)
        assert_eq(effects.overlaps(), execution == OverlappedTransfer)
        assert_eq(
          effects.pending_group_limit(),
          if execution == OverlappedTransfer {
            stages - 2
          } else {
            0
          },
        )
        assert_true(effects.prime_count() <= transfers)
        assert_true(effects.pending_group_limit() < effects.prime_count())
        let slots : Array[Int?] = Array::make(stages, None)
        for prime in 0..<effects.prime_count() {
          slots[plan.slot(prime).unwrap()] = Some(prime)
        }
        for transfer in 0..<transfers {
          let current = plan.slot(transfer).unwrap()
          let future = transfer + effects.prime_count()
          let mut consumed = false
          let mut issued = false
          let mut waited = false
          let mut published = false
          for action in effects.iteration() {
            assert_false(published)
            match action {
              IssueFutureOperand => {
                assert_false(issued)
                assert_eq(consumed, execution == SerialTransfer)
                if future < transfers {
                  let destination = plan.slot(future).unwrap()
                  assert_false(destination == current)
                  assert_eq(slots[destination], None)
                  slots[destination] = Some(future)
                }
                issued = true
              }
              ConsumeCurrentOperand => {
                assert_false(consumed)
                assert_eq(slots[current], Some(transfer))
                slots[current] = None
                consumed = true
              }
              AwaitOperandTransfers => {
                assert_true(issued && consumed)
                assert_false(waited)
                waited = true
              }
              PublishOperandSlot => {
                assert_true(consumed && issued)
                assert_eq(waited, execution == OverlappedTransfer)
                published = true
              }
            }
          }
          assert_true(published)
        }
        assert_true(slots.all(fn(slot) { slot == None }))
      }
      assert_eq(plan.slot(-1), None)
      assert_eq(plan.slot(transfers), None)
    }
  }
}
