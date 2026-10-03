///|
fn exercise_independent_ring(
  tiles : Int,
  invalid_epoch : Int?,
  invalid_value : Bool,
  omit_key_wait? : Bool = false,
  omit_key_release? : Bool = false,
) -> Unit raise {
  let plan = @luna_attention_tile_schedule.AttentionGroupedEffectPlan::new(
    AttentionAsyncStagedShared,
    stages=2,
    readiness=KeyScoreBeforeValueReadiness,
    independent=true,
  )
  let machine = QueryEffectMachine::new(2, true)
  let issue = (operand : Int, slot : Int, epoch : Int) => {
    // Out-of-domain groups are empty, not omitted. This includes an empty
    // partition, one-tile partitions and both trailing refills.
    machine.issue(
      if epoch < tiles {
        [(slot * 2 + operand, epoch)]
      } else {
        []
      },
    )
  }
  let apply = (
    point : @luna_attention_tile_schedule.AttentionGroupedEffectPoint,
    epoch : Int,
  ) => {
    for action in plan.actions(point) {
      match action {
        ResetOperandValidity => ()
        PrimeOperandRings => {
          issue(0, 0, 0)
          issue(1, 0, 0)
          issue(0, 1, 1)
          issue(1, 1, 1)
        }
        AwaitPendingTransfers(groups) =>
          if !(omit_key_wait && point == AcquireKeyOperands) {
            machine.apply([AwaitPendingTransfers(groups)], epoch, false)
          }
        PublishWorkgroup => machine.apply([PublishWorkgroup], epoch, false)
        RequireValidKeyStage | RequireValidValueStage => ()
        RefillKeyStage => issue(0, machine.slot, epoch + 2)
        RefillValueStage => issue(1, machine.slot, epoch + 2)
        _ => fail("unexpected pair transfer in independent ring")
      }
    }
  }
  apply(InitializeTransfers, 0)
  for epoch in 0..<tiles {
    machine.slot = epoch % 2
    let key = machine.slot * 2
    apply(AcquireKeyOperands, epoch)
    assert_eq(machine.visible[key], epoch, msg="QK read incomplete K")
    if invalid_epoch == Some(epoch) && !invalid_value {
      apply(CompleteTileReaders, epoch)
      assert_true(machine.pending.is_empty())
      return
    }
    if !omit_key_release {
      machine.consumed[key] = epoch
    }
    apply(ReleaseKeyReaders, epoch)
    // K for epoch+2 is pending while V for this epoch becomes visible.
    // Priming only one stage or waiting zero here defeats the pipeline.
    assert_eq(machine.pending.length(), 4)
    apply(AcquireValueOperands, epoch)
    assert_eq(machine.visible[key + 1], epoch, msg="PV read incomplete V")
    if invalid_epoch == Some(epoch) && invalid_value {
      apply(CompleteTileReaders, epoch)
      assert_true(machine.pending.is_empty())
      return
    }
    machine.consumed[key + 1] = epoch
    apply(ReleaseTileReaders, epoch)
    assert_eq(machine.pending.length(), 4)
  }
  apply(CompleteTileReaders, tiles)
  assert_true(machine.pending.is_empty())
  for index in 0..<4 {
    assert_eq(machine.ready[index], machine.consumed[index])
  }
}

///|
test "independent two-stage decode preserves both operand lifetimes including tails" {
  for tiles in 0..<=11 {
    exercise_independent_ring(tiles, None, false)
    for invalid in 0..<tiles {
      for value in [false, true] {
        exercise_independent_ring(tiles, Some(invalid), value)
      }
    }
  }
}

///|
test "independent decode model rejects absent key wait and early key overwrite" {
  for defect in 0..<2 {
    try
      exercise_independent_ring(
        4,
        None,
        false,
        omit_key_wait=defect == 0,
        omit_key_release=defect == 1,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("accepted invalid independent K lifetime")
    }
  }
}

///|
test "independent decode plan rejects single-slot storage" {
  try
    @luna_attention_tile_schedule.AttentionGroupedEffectPlan::new(
      AttentionStagedShared,
      stages=1,
      readiness=JoinedKeyValueReadiness,
      independent=true,
    )
  catch {
    _ => ()
  } noraise {
    _ => fail("accepted single-slot independent pipeline")
  }
}
