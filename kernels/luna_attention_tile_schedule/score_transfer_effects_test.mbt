///|
priv struct ScoreValidationLifetime {
  mut readers_live : Bool
  mut visible : Bool
}

///|
fn apply_score_transfers(
  machine : QueryEffectMachine,
  lifetime : ScoreValidationLifetime,
  actions : ArrayView[
    @luna_attention_tile_schedule.AttentionScoreTransferEffect,
  ],
  epoch : Int,
  tiles : Int,
  invalid : Bool,
  defect : Int,
) -> Bool raise {
  let key = machine.slot * 2
  for action in actions {
    match action {
      ResetValidity => {
        assert_false(lifetime.readers_live)
        lifetime.visible = false
      }
      PublishWorkgroup =>
        if defect != 2 {
          machine.apply([PublishWorkgroup], epoch, false)
          lifetime.visible = true
        }
      StageCurrentPair => {
        assert_false(lifetime.readers_live)
        machine.issue([(key, epoch), (key + 1, epoch)])
        lifetime.visible = false
      }
      AwaitPendingTransfers =>
        if defect != 1 {
          machine.apply([AwaitPendingTransfers(0)], epoch, false)
        }
      RequireValidTile => {
        assert_true(
          machine.pending.is_empty(),
          msg="early exit with live writer",
        )
        assert_true(lifetime.visible, msg="validation read before publication")
        if invalid {
          return false
        }
        lifetime.readers_live = true
      }
      ReleaseValidationReaders =>
        if defect != 3 {
          machine.apply([PublishWorkgroup], epoch, false)
          lifetime.readers_live = false
          lifetime.visible = true
        }
      StageNextPairInOtherSlot =>
        if epoch + 1 < tiles {
          assert_false(
            lifetime.readers_live,
            msg="prefetch races validity readers",
          )
          let next = (machine.slot + 1) % machine.stages * 2
          machine.issue([(next, epoch + 1), (next + 1, epoch + 1)])
          lifetime.visible = false
        }
      StageFirstPair =>
        if tiles > 0 {
          machine.issue([(0, 0), (1, 0)])
          lifetime.visible = false
        }
    }
  }
  true
}

///|
fn exercise_score_transfers(
  mode : @luna_attention_tile_schedule.AttentionScoreTransferMode,
  stages : Int,
  tiles : Int,
  invalid_epoch : Int?,
  defect? : Int = 0,
) -> Unit raise {
  let machine = QueryEffectMachine::new(stages, mode != CooperativeTransfer)
  let lifetime = ScoreValidationLifetime::{
    readers_live: false,
    visible: true,
  }
  ignore(
    apply_score_transfers(
      machine,
      lifetime,
      mode.actions(PrimeScoreFold),
      0,
      tiles,
      false,
      defect,
    ),
  )
  for epoch in 0..<tiles {
    machine.slot = epoch % stages
    let key = machine.slot * 2
    if !apply_score_transfers(
        machine,
        lifetime,
        mode.actions(AcquireScoreTile),
        epoch,
        tiles,
        invalid_epoch == Some(epoch),
        defect,
      ) {
      return
    }
    // QK may alias validation storage, and PV consumes the same tile's V.
    assert_false(lifetime.readers_live, msg="score write races validation read")
    assert_eq(machine.visible[key], epoch, msg="QK read before publication")
    assert_eq(machine.visible[key + 1], epoch, msg="PV read before publication")
    if defect != 4 {
      assert_eq(mode.actions(ReleaseScoreTile).iter().to_array(), [
        PublishWorkgroup,
      ])
      ignore(
        apply_score_transfers(
          machine,
          lifetime,
          mode.actions(ReleaseScoreTile),
          epoch,
          tiles,
          false,
          defect,
        ),
      )
      machine.consumed[key] = epoch
      machine.consumed[key + 1] = epoch
    }
  }
  assert_true(machine.pending.is_empty())
  for i in 0..<machine.ready.length() {
    assert_eq(machine.consumed[i], machine.ready[i])
  }
}

///|
test "shared score transfer lifetimes cover all rings tails and invalid exits" {
  for
    (memory, stages, immediate) in [
      (@luna_attention_strategy.AttentionStagedShared, 1, false),
      (AttentionStagedShared, 2, false),
      (AttentionStagedShared, 3, false),
      (AttentionStagedShared, 4, false),
      (AttentionStagedShared, 1, true),
      (AttentionAsyncStagedShared, 2, false),
    ] {
    let mode = @luna_attention_tile_schedule.AttentionScoreTransferMode::from_schedule(
      memory,
      stages~,
      immediate~,
    )
    for tiles in 0..<=9 {
      exercise_score_transfers(mode, stages, tiles, None)
      for invalid in 0..<tiles {
        exercise_score_transfers(mode, stages, tiles, Some(invalid))
      }
    }
  }
}

///|
test "shared score transfer model catches omitted lifetime edges" {
  for
    (mode, stages) in [
      (@luna_attention_tile_schedule.ImmediateTransfer, 1),
      (LookaheadTransfer, 2),
    ] {
    for defect in 1..<=4 {
      try exercise_score_transfers(mode, stages, 5, None, defect~) catch {
        _ => ()
      } noraise {
        _ => fail("accepted missing wait/publication/validation/read release")
      }
    }
  }
}

///|
test "shared score transfer rejects unsupported rings" {
  for
    memory in [
      @luna_attention_strategy.AttentionDirectGlobal,
      AttentionStagedShared,
      AttentionAsyncStagedShared,
    ] {
    for stages in 0..<=5 {
      for immediate in [false, true] {
        let legal = (
            memory == AttentionStagedShared &&
            (
              (!immediate && stages >= 1 && stages <= 4) ||
              (immediate && stages == 1)
            )
          ) ||
          (memory == AttentionAsyncStagedShared && stages == 2 && !immediate)
        try
          @luna_attention_tile_schedule.AttentionScoreTransferMode::from_schedule(
            memory,
            stages~,
            immediate~,
          )
        catch {
          _ => assert_false(legal)
        } noraise {
          _ => assert_true(legal)
        }
      }
    }
  }
}
