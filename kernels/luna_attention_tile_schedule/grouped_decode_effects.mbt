///|
/// Workgroup-uniform ownership boundaries for a grouped ordered KV fold.
/// Query/score arithmetic is pure between these effect boundaries.
pub(all) enum AttentionGroupedEffectPoint {
  InitializeTransfers
  AcquireKeyOperands
  AcquireValueOperands
  ReleaseKeyReaders
  PrefetchNextTile
  ReleaseTileReaders
  CompleteTileReaders
  PublishFoldPartials
} derive(Debug, Eq)

///|
/// Logical effects; the backend supplies transfer addresses and instructions.
/// Requiring validity is a uniform early exit after all relevant transfers
/// have completed, never an exit with a live asynchronous writer.
pub(all) enum AttentionGroupedEffect {
  ResetStageValidity
  ResetTileValidity
  PublishWorkgroup
  StageFirstPair
  StageCurrentPair
  AwaitPendingTransfers(Int)
  RequireValidTile
  RequireValidStage
  StageNextPairInOtherSlot
  ResetOperandValidity
  PrimeOperandRings
  RequireValidKeyStage
  RequireValidValueStage
  RefillKeyStage
  RefillValueStage
} derive(Debug, Eq)

///|
pub struct AttentionGroupedEffectPlan {
  priv pipelined : Bool
  priv independent : Bool
} derive(Debug, Eq)

///|
/// Cooperative single-slot and split-readiness double-slot are the existing
/// supported grouped folds. No target or model name participates in selection.
pub fn AttentionGroupedEffectPlan::new(
  memory : @luna_attention_strategy.AttentionTileMemory,
  stages~ : Int,
  readiness~ : AttentionTransferReadiness,
  independent? : Bool = false,
) -> AttentionGroupedEffectPlan raise AttentionTileScheduleError {
  let pipelined = match (memory, stages, readiness) {
    (AttentionStagedShared, 1, JoinedKeyValueReadiness) => false
    (AttentionAsyncStagedShared, 2, KeyScoreBeforeValueReadiness) => true
    _ => raise InvalidAttentionTileSchedule(InvalidLoopNest)
  }
  guard !independent || pipelined else {
    raise InvalidAttentionTileSchedule(InvalidLoopNest)
  }
  { pipelined, independent, }
}

///|
pub fn AttentionGroupedEffectPlan::independently_refills(
  self : AttentionGroupedEffectPlan,
) -> Bool {
  self.independent
}

///|
pub fn AttentionGroupedEffectPlan::is_pipelined(
  self : AttentionGroupedEffectPlan,
) -> Bool {
  self.pipelined
}

///|
pub fn AttentionGroupedEffectPlan::actions(
  self : AttentionGroupedEffectPlan,
  point : AttentionGroupedEffectPoint,
) -> ReadOnlyArray[AttentionGroupedEffect] {
  if self.independent {
    return match point {
      InitializeTransfers =>
        [ResetOperandValidity, PublishWorkgroup, PrimeOperandRings]
      // Four committed groups are always present: K0,V0,K1,V1. Each
      // reader waits only for the oldest group, then retires its own slot.
      // Empty out-of-domain copies still commit a group, preserving the law.
      AcquireKeyOperands =>
        [AwaitPendingTransfers(3), PublishWorkgroup, RequireValidKeyStage]
      ReleaseKeyReaders => [PublishWorkgroup, RefillKeyStage]
      AcquireValueOperands =>
        [AwaitPendingTransfers(3), PublishWorkgroup, RequireValidValueStage]
      ReleaseTileReaders => [PublishWorkgroup, RefillValueStage]
      CompleteTileReaders => [AwaitPendingTransfers(0), PublishWorkgroup]
      PublishFoldPartials => [PublishWorkgroup]
      PrefetchNextTile => []
    }
  }
  match (point, self.pipelined) {
    (InitializeTransfers, true) =>
      [ResetStageValidity, PublishWorkgroup, StageFirstPair]
    (AcquireKeyOperands, true) => [AwaitPendingTransfers(1), PublishWorkgroup]
    (AcquireKeyOperands, false) =>
      [
        ResetTileValidity,
        PublishWorkgroup,
        StageCurrentPair,
        PublishWorkgroup,
        RequireValidTile,
      ]
    (AcquireValueOperands, true) =>
      [AwaitPendingTransfers(0), PublishWorkgroup, RequireValidStage]
    (PrefetchNextTile, true) => [StageNextPairInOtherSlot]
    // In the double-slot fold the following key publication also ends the
    // previous readers' epoch, before PrefetchNextTile reuses their slot.
    // At loop exit there is no next acquire: publish before arena reuse.
    (ReleaseTileReaders, false) => [PublishWorkgroup]
    (CompleteTileReaders, true) => [PublishWorkgroup]
    (PublishFoldPartials, _) => [PublishWorkgroup]
    _ => []
  }
}
