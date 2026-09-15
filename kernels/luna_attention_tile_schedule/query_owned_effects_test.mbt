///|
/// Test-only worst-case asynchronous memory model: an issued group remains
/// invisible until a wait completes it, and completed writes become visible
/// to consumers only at publication. No device execution is simulated here.
priv struct QueryEffectMachine {
  ready : Array[Int]
  visible : Array[Int]
  consumed : Array[Int]
  pending : Array[Array[(Int, Int)]]
  mut slot : Int
  stages : Int
  asynchronous : Bool
}

///|
fn QueryEffectMachine::new(
  stages : Int,
  asynchronous : Bool,
) -> QueryEffectMachine {
  {
    ready: Array::make(stages * 2, -1),
    visible: Array::make(stages * 2, -1),
    consumed: Array::make(stages * 2, -1),
    pending: [],
    slot: 0,
    stages,
    asynchronous,
  }
}

///|
fn QueryEffectMachine::issue(
  self : QueryEffectMachine,
  writes : Array[(Int, Int)],
) -> Unit raise {
  for (index, _) in writes {
    assert_true(
      self.consumed[index] >= self.ready[index],
      msg="overwrite before last read",
    )
    assert_false(self.pending.any(group => group.any(w => w.0 == index)))
  }
  if self.asynchronous {
    self.pending.push(writes)
  } else {
    for (index, epoch) in writes {
      self.ready[index] = epoch
    }
  }
}

///|
fn QueryEffectMachine::apply(
  self : QueryEffectMachine,
  actions : ArrayView[@luna_attention_tile_schedule.AttentionQueryEffect],
  epoch : Int,
  has_next : Bool,
) -> Unit raise {
  let key = self.slot * 2
  for action in actions {
    match action {
      StageFirstKey => self.issue([(0, 0)])
      StageFirstValue => self.issue([(1, 0)])
      StageFirstPair => self.issue([(0, 0), (1, 0)])
      StageCurrentPair => self.issue([(key, epoch), (key + 1, epoch)])
      AwaitPendingTransfers(remaining) => {
        assert_true(remaining >= 0)
        while self.pending.length() > remaining {
          let group = self.pending.remove(0)
          for (index, generation) in group {
            self.ready[index] = generation
          }
        }
      }
      PublishWorkgroup =>
        for i in 0..<self.ready.length() {
          self.visible[i] = self.ready[i]
        }
      StageNextPairInOtherSlot =>
        if has_next {
          let next = (self.slot + 1) % self.stages * 2
          self.issue([(next, epoch + 1), (next + 1, epoch + 1)])
        }
      StageNextKey => if has_next { self.issue([(key, epoch + 1)]) }
      StageNextValue => if has_next { self.issue([(key + 1, epoch + 1)]) }
      AdvanceTransferSlot => self.slot = (self.slot + 1) % self.stages
    }
  }
}

///|
fn exercise_query_effects(
  plan : @luna_attention_tile_schedule.AttentionQueryEffectPlan,
  stages : Int,
  asynchronous : Bool,
  tiles : Int,
  invalid_epoch : Int?,
) -> Unit raise {
  let machine = QueryEffectMachine::new(stages, asynchronous)
  machine.apply(plan.actions(PublishQueryOperands), 0, false)
  if tiles > 0 {
    machine.apply(plan.actions(PrimeQueryFold), 0, false)
  }
  for epoch in 0..<tiles {
    let next = epoch + 1 < tiles
    let key = machine.slot * 2
    let points : Array[@luna_attention_tile_schedule.AttentionQueryEffectPoint] = [
      IssueCurrentKeyValue,
      AwaitKeyReadiness,
      PublishKeyReadiness,
    ]
    for point in points {
      machine.apply(plan.actions(point), epoch, next)
    }
    if invalid_epoch == Some(epoch) {
      machine.apply(plan.actions(InvalidQueryTileExit), epoch, next)
      assert_true(
        machine.pending.is_empty(),
        msg="exit left an in-flight transfer",
      )
      return
    }
    assert_eq(machine.visible[key], epoch, msg="QK read unpublished K")
    machine.apply(plan.actions(AfterKeyPublication), epoch, next)
    machine.consumed[key] = epoch
    machine.apply(plan.actions(BeforeValueComputation), epoch, next)
    assert_eq(machine.visible[key + 1], epoch, msg="PV read unpublished V")
    machine.consumed[key + 1] = epoch
    machine.apply(plan.actions(ReleaseValueReaders), epoch, next)
    machine.apply(plan.actions(AfterValueRelease), epoch, next)
  }
  machine.apply(plan.actions(ReleaseQueryReaders), tiles, false)
  assert_true(machine.pending.is_empty())
  for i in 0..<machine.ready.length() {
    assert_eq(machine.consumed[i], machine.ready[i])
  }
}

///|
test "query effects preserve readiness and reuse across tails and error exits" {
  for
    (memory, stages, readiness) in [
      (
        @luna_attention_strategy.AttentionStagedShared,
        1,
        @luna_attention_tile_schedule.JoinedKeyValueReadiness,
      ),
      (AttentionAsyncStagedShared, 1, KeyScoreBeforeValueReadiness),
      (AttentionAsyncStagedShared, 2, JoinedKeyValueReadiness),
    ] {
    let plan = @luna_attention_tile_schedule.AttentionQueryEffectPlan::new(
      memory,
      stages~,
      readiness~,
    )
    for tiles in 0..<=8 {
      exercise_query_effects(
        plan,
        stages,
        memory == AttentionAsyncStagedShared,
        tiles,
        None,
      )
      for invalid in 0..<tiles {
        exercise_query_effects(
          plan,
          stages,
          memory == AttentionAsyncStagedShared,
          tiles,
          Some(invalid),
        )
      }
    }
  }
}

///|
test "query effect model detects missing readiness and premature reuse" {
  let machine = QueryEffectMachine::new(1, true)
  machine.apply([StageFirstKey, StageFirstValue, PublishWorkgroup], 0, false)
  assert_eq(machine.visible[0], -1)
  machine.apply([AwaitPendingTransfers(1), PublishWorkgroup], 0, false)
  assert_eq(machine.visible[0], 0)
  assert_eq(machine.visible[1], -1)
  try machine.apply([StageNextKey], 0, true) catch {
    _ => ()
  } noraise {
    _ => fail("overwrote an unconsumed K tile")
  }
  machine.apply([AwaitPendingTransfers(0), PublishWorkgroup], 0, false)
  assert_eq(machine.visible[1], 0)
}

///|
test "split lifetime publishes value before recycling key storage" {
  let plan = @luna_attention_tile_schedule.AttentionQueryEffectPlan::new(
    @luna_attention_strategy.AttentionAsyncStagedShared,
    stages=1,
    readiness=KeyScoreBeforeValueReadiness,
  )
  assert_eq(plan.actions(BeforeValueComputation).iter().to_array(), [
    AwaitPendingTransfers(0),
    PublishWorkgroup,
    StageNextKey,
  ])
  assert_eq(plan.actions(ReleaseValueReaders).iter().to_array(), [
    PublishWorkgroup,
  ])
  assert_eq(plan.actions(AfterValueRelease).iter().to_array(), [StageNextValue])
  // Future K is completed at the next iteration's readiness point; release
  // alone publishes only completed writes, never an outstanding transfer.
  assert_eq(plan.actions(AwaitKeyReadiness).iter().to_array(), [
    AwaitPendingTransfers(1),
  ])
  assert_eq(plan.actions(PublishKeyReadiness).iter().to_array(), [
    PublishWorkgroup,
  ])
}

///|
test "query effects reject unsupported lifetime combinations" {
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
            (
              (stages == 1 && readiness == KeyScoreBeforeValueReadiness) ||
              (stages == 2 && readiness == JoinedKeyValueReadiness)
            )
          )
        try
          @luna_attention_tile_schedule.AttentionQueryEffectPlan::new(
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
