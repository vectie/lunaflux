///|
test "enclosing operand publications subsume only redundant blockwise exchanges" {
  let fold = @attention_physical_ir.BlockwiseFold::new(
    dimensions=128,
    keys=32,
    score_owners=2,
  ).unwrap()
  let arithmetic : Array[@attention_physical_ir.BlockwiseFoldStep] = [
    ComputeTileScores,
    ReduceTileMaximum,
    FormFloat32Probabilities,
    ReduceTileDenominator,
    AccumulateTileValues,
    MergeTileState,
  ]
  let variants : Array[
    (
      @luna_attention_strategy.AttentionTileMemory,
      Int,
      @luna_attention_tile_schedule.AttentionTransferReadiness,
      Bool,
      Bool,
      Bool,
    ),
  ] = [
    (AttentionStagedShared, 1, JoinedKeyValueReadiness, false, true, true),
    (
      AttentionAsyncStagedShared,
      2,
      KeyScoreBeforeValueReadiness,
      false,
      true,
      false,
    ),
    (
      AttentionAsyncStagedShared,
      2,
      KeyScoreBeforeValueReadiness,
      true,
      false,
      false,
    ),
  ]
  for (memory, stages, readiness, independent, score, probability) in variants {
    let plan = @luna_attention_tile_schedule.AttentionGroupedEffectPlan::new(
      memory,
      stages~,
      readiness~,
      independent~,
    )
    let steps = plan.blockwise_steps(fold)
    assert_eq(steps.contains(PublishTileScores), score)
    assert_eq(steps.contains(PublishTileProbabilities), probability)
    assert_eq(
      steps
      .filter(step => {
        step != PublishTileScores && step != PublishTileProbabilities
      })
      .iter()
      .to_array(),
      arithmetic,
    )
    if !score {
      assert_true(plan.actions(ReleaseKeyReaders).contains(PublishWorkgroup))
    }
    if !probability {
      assert_true(plan.actions(AcquireValueOperands).contains(PublishWorkgroup))
    }
  }
}
