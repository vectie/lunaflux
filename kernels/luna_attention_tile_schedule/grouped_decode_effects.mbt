///|
/// Workgroup-uniform ownership boundaries for a grouped ordered KV fold.
/// Query/score arithmetic is pure between these effect boundaries.
pub(all) enum AttentionGroupedEffectPoint {
  InitializeTransfers
  AcquireKeyOperands
  AcquireValueOperands
  PrefetchNextTile
  ReleaseTileReaders
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
} derive(Debug, Eq)

///|
pub struct AttentionGroupedEffectPlan {
  priv pipelined : Bool
} derive(Debug, Eq)

///|
/// Cooperative single-slot and split-readiness double-slot are the existing
/// supported grouped folds. No target or model name participates in selection.
pub fn AttentionGroupedEffectPlan::new(
  memory : @luna_attention_strategy.AttentionTileMemory,
  stages~ : Int,
  readiness~ : AttentionTransferReadiness,
) -> AttentionGroupedEffectPlan raise AttentionTileScheduleError {
  let pipelined = match (memory, stages, readiness) {
    (AttentionStagedShared, 1, JoinedKeyValueReadiness) => false
    (AttentionAsyncStagedShared, 2, KeyScoreBeforeValueReadiness) => true
    _ => raise InvalidAttentionTileSchedule(InvalidLoopNest)
  }
  { pipelined, }
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
    (ReleaseTileReaders | PublishFoldPartials, _) => [PublishWorkgroup]
    _ => []
  }
}
