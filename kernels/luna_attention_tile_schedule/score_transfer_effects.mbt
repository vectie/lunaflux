///|
/// Transport lifetime for shared-score QK/softmax/PV. Immediate transfer uses
/// asynchronous transport but has no lookahead and no live writer during QK.
pub(all) enum AttentionScoreTransferMode {
  CooperativeTransfer
  ImmediateTransfer
  LookaheadTransfer
} derive(Debug, Eq)

///|
pub(all) enum AttentionScoreTransferPoint {
  PrimeScoreFold
  AcquireScoreTile
  ReleaseScoreTile
} derive(Debug, Eq)

///|
pub(all) enum AttentionScoreTransferEffect {
  ResetValidity
  PublishWorkgroup
  StageCurrentPair
  AwaitPendingTransfers
  RequireValidTile
  ReleaseValidationReaders
  StageNextPairInOtherSlot
  StageFirstPair
} derive(Debug, Eq)

///|
pub fn AttentionScoreTransferMode::from_schedule(
  memory : @luna_attention_strategy.AttentionTileMemory,
  stages~ : Int,
  immediate~ : Bool,
) -> AttentionScoreTransferMode raise AttentionTileScheduleError {
  match (memory, stages, immediate) {
    (AttentionStagedShared, 1, true) => ImmediateTransfer
    (AttentionStagedShared, 1..=4, false) => CooperativeTransfer
    (AttentionAsyncStagedShared, 2, false) => LookaheadTransfer
    _ => raise InvalidAttentionTileSchedule(InvalidLoopNest)
  }
}

///|
/// Validation reads must end before the score slot can be reused or the next
/// transfer can update validity. Reader release likewise precedes ring reuse.
pub fn AttentionScoreTransferMode::actions(
  self : AttentionScoreTransferMode,
  point : AttentionScoreTransferPoint,
) -> ReadOnlyArray[AttentionScoreTransferEffect] {
  match (point, self) {
    (PrimeScoreFold, LookaheadTransfer) => [StageFirstPair]
    (PrimeScoreFold, _) => []
    (AcquireScoreTile, LookaheadTransfer) =>
      [
        AwaitPendingTransfers,
        PublishWorkgroup,
        RequireValidTile,
        ReleaseValidationReaders,
        StageNextPairInOtherSlot,
      ]
    (AcquireScoreTile, ImmediateTransfer) =>
      [
        ResetValidity,
        PublishWorkgroup,
        StageCurrentPair,
        AwaitPendingTransfers,
        PublishWorkgroup,
        RequireValidTile,
        ReleaseValidationReaders,
      ]
    (AcquireScoreTile, CooperativeTransfer) =>
      [
        ResetValidity,
        PublishWorkgroup,
        StageCurrentPair,
        PublishWorkgroup,
        RequireValidTile,
        ReleaseValidationReaders,
      ]
    (ReleaseScoreTile, _) => [PublishWorkgroup]
  }
}
