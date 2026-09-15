///|
fn apply_grouped_effects(
  machine : QueryEffectMachine,
  actions : ArrayView[@luna_attention_tile_schedule.AttentionGroupedEffect],
  epoch : Int,
  tiles : Int,
  invalid : Bool,
  omit_wait : Bool,
  omit_publication : Bool,
) -> Bool raise {
  let key = machine.slot * 2
  for action in actions {
    match action {
      ResetStageValidity | ResetTileValidity => ()
      PublishWorkgroup =>
        if !omit_publication {
          machine.apply([PublishWorkgroup], epoch, false)
        }
      StageFirstPair =>
        if tiles > 0 {
          // The physical transfer has two independently committed groups.
          machine.issue([(0, 0)])
          machine.issue([(1, 0)])
        }
      StageCurrentPair => machine.issue([(key, epoch), (key + 1, epoch)])
      AwaitPendingTransfers(remaining) =>
        if !omit_wait {
          machine.apply([AwaitPendingTransfers(remaining)], epoch, false)
        }
      RequireValidTile | RequireValidStage => {
        assert_true(
          machine.pending.is_empty(),
          msg="exit before transfer drain",
        )
        if invalid {
          return false
        }
      }
      StageNextPairInOtherSlot =>
        if epoch + 1 < tiles {
          let next = (machine.slot + 1) % machine.stages * 2
          machine.issue([(next, epoch + 1)])
          machine.issue([(next + 1, epoch + 1)])
        }
    }
  }
  true
}

///|
fn exercise_grouped_effects(
  plan : @luna_attention_tile_schedule.AttentionGroupedEffectPlan,
  tiles : Int,
  invalid_epoch : Int?,
  omit_wait? : Bool = false,
  omit_publication? : Bool = false,
  omit_release? : Bool = false,
) -> Unit raise {
  let machine = QueryEffectMachine::new(
    if plan.is_pipelined() {
      2
    } else {
      1
    },
    plan.is_pipelined(),
  )
  ignore(
    apply_grouped_effects(
      machine,
      plan.actions(InitializeTransfers),
      0,
      tiles,
      false,
      omit_wait,
      omit_publication,
    ),
  )
  for epoch in 0..<tiles {
    machine.slot = epoch % machine.stages
    let key = machine.slot * 2
    if !apply_grouped_effects(
        machine,
        plan.actions(AcquireKeyOperands),
        epoch,
        tiles,
        invalid_epoch == Some(epoch),
        omit_wait,
        omit_publication,
      ) {
      return
    }
    assert_eq(machine.visible[key], epoch, msg="QK read an unpublished key")
    if !apply_grouped_effects(
        machine,
        plan.actions(AcquireValueOperands),
        epoch,
        tiles,
        invalid_epoch == Some(epoch),
        omit_wait,
        omit_publication,
      ) {
      return
    }
    ignore(
      apply_grouped_effects(
        machine,
        plan.actions(PrefetchNextTile),
        epoch,
        tiles,
        false,
        omit_wait,
        omit_publication,
      ),
    )
    assert_eq(
      machine.visible[key + 1],
      epoch,
      msg="PV read an unpublished value",
    )
    if !omit_release {
      assert_eq(plan.actions(ReleaseTileReaders).iter().to_array(), [
        PublishWorkgroup,
      ])
      ignore(
        apply_grouped_effects(
          machine,
          plan.actions(ReleaseTileReaders),
          epoch,
          tiles,
          false,
          omit_wait,
          omit_publication,
        ),
      )
      // All workers have completed both reads before this slot is reusable.
      machine.consumed[key] = epoch
      machine.consumed[key + 1] = epoch
    }
  }
  assert_true(machine.pending.is_empty())
  for i in 0..<machine.ready.length() {
    assert_eq(machine.consumed[i], machine.ready[i])
  }
  assert_eq(plan.actions(PublishFoldPartials).iter().to_array(), [
    PublishWorkgroup,
  ])
}

///|
test "grouped fold effects drain errors and release both ring slots" {
  for
    (memory, stages, readiness) in [
      (
        @luna_attention_strategy.AttentionStagedShared,
        1,
        @luna_attention_tile_schedule.JoinedKeyValueReadiness,
      ),
      (AttentionAsyncStagedShared, 2, KeyScoreBeforeValueReadiness),
    ] {
    let plan = @luna_attention_tile_schedule.AttentionGroupedEffectPlan::new(
      memory,
      stages~,
      readiness~,
    )
    for tiles in 0..<=9 {
      exercise_grouped_effects(plan, tiles, None)
      for invalid in 0..<tiles {
        exercise_grouped_effects(plan, tiles, Some(invalid))
      }
    }
  }
}

///|
test "grouped effect model rejects missing waits publication and release" {
  let plan = @luna_attention_tile_schedule.AttentionGroupedEffectPlan::new(
    @luna_attention_strategy.AttentionAsyncStagedShared,
    stages=2,
    readiness=KeyScoreBeforeValueReadiness,
  )
  for defect in 0..<3 {
    try
      exercise_grouped_effects(
        plan,
        4,
        None,
        omit_wait=defect == 0,
        omit_publication=defect == 1,
        omit_release=defect == 2,
      )
    catch {
      _ => ()
    } noraise {
      _ => fail("accepted an invalid grouped lifetime")
    }
  }
  assert_eq(plan.actions(AcquireValueOperands).iter().to_array(), [
    AwaitPendingTransfers(0),
    PublishWorkgroup,
    RequireValidStage,
  ])
  assert_eq(plan.actions(PrefetchNextTile).iter().to_array(), [
    StageNextPairInOtherSlot,
  ])
}

///|
test "grouped plan rejects unimplemented lifetime combinations" {
  for
    memory in [
      @luna_attention_strategy.AttentionDirectGlobal,
      AttentionStagedShared,
      AttentionAsyncStagedShared,
    ] {
    for stages in [0, 1, 2, 3] {
      for
        readiness in [
          @luna_attention_tile_schedule.JoinedKeyValueReadiness,
          KeyScoreBeforeValueReadiness,
        ] {
        let legal = (
            memory == AttentionStagedShared &&
            stages == 1 &&
            readiness == JoinedKeyValueReadiness
          ) ||
          (
            memory == AttentionAsyncStagedShared &&
            stages == 2 &&
            readiness == KeyScoreBeforeValueReadiness
          )
        try
          @luna_attention_tile_schedule.AttentionGroupedEffectPlan::new(
            memory,
            stages~,
            readiness~,
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
