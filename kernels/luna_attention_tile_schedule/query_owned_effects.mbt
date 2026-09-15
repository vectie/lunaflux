///|
/// Effect boundaries around the ordered QK and softmax/PV computations.
/// All participating workers traverse these boundaries, including inactive
/// query owners. The arithmetic inside either computation is unchanged.
pub(all) enum AttentionQueryEffectPoint {
  PublishQueryOperands
  PrimeQueryFold
  IssueCurrentKeyValue
  AwaitKeyReadiness
  PublishKeyReadiness
  InvalidQueryTileExit
  AfterKeyPublication
  BeforeValueComputation
  ReleaseValueReaders
  AfterValueRelease
  ReleaseQueryReaders
} derive(Debug, Eq)

///|
/// Logical transfer and ownership effects. Future transfers are conditional
/// on another KV tile existing; waits and publications are workgroup-uniform.
/// No action names a vendor instruction, lane width or physical address.
pub(all) enum AttentionQueryEffect {
  StageFirstKey
  StageFirstValue
  StageFirstPair
  StageCurrentPair
  AwaitPendingTransfers(Int)
  PublishWorkgroup
  StageNextPairInOtherSlot
  StageNextKey
  StageNextValue
  AdvanceTransferSlot
} derive(Debug, Eq)

///|
priv enum QueryTransferLifetime {
  CooperativeSingleSlot
  SplitSingleSlot
  JoinedDoubleSlot
} derive(Debug, Eq)

///|
pub struct AttentionQueryEffectPlan {
  priv lifetime : QueryTransferLifetime
} derive(Debug, Eq)

///|
/// Derive effects from the immutable schedule, rather than deciding issue,
/// wait and buffer-reuse order while emitting device source.
pub fn AttentionQueryEffectPlan::new(
  memory : @luna_attention_strategy.AttentionTileMemory,
  stages~ : Int,
  readiness~ : AttentionTransferReadiness,
) -> AttentionQueryEffectPlan raise AttentionTileScheduleError {
  let lifetime = match (memory, stages, readiness) {
    (AttentionStagedShared, 1, JoinedKeyValueReadiness) => CooperativeSingleSlot
    (AttentionAsyncStagedShared, 1, KeyScoreBeforeValueReadiness) =>
      SplitSingleSlot
    (AttentionAsyncStagedShared, 2, JoinedKeyValueReadiness) => JoinedDoubleSlot
    _ => raise InvalidAttentionTileSchedule(InvalidLoopNest)
  }
  { lifetime, }
}

///|
pub fn AttentionQueryEffectPlan::split_key_value(
  self : AttentionQueryEffectPlan,
) -> Bool {
  self.lifetime == SplitSingleSlot
}

///|
pub fn AttentionQueryEffectPlan::actions(
  self : AttentionQueryEffectPlan,
  point : AttentionQueryEffectPoint,
) -> ReadOnlyArray[AttentionQueryEffect] {
  match (point, self.lifetime) {
    (PrimeQueryFold, SplitSingleSlot) => [StageFirstKey, StageFirstValue]
    (PrimeQueryFold, JoinedDoubleSlot) => [StageFirstPair]
    (IssueCurrentKeyValue, CooperativeSingleSlot) => [StageCurrentPair]
    (AwaitKeyReadiness, SplitSingleSlot) => [AwaitPendingTransfers(1)]
    (AwaitKeyReadiness, JoinedDoubleSlot) => [AwaitPendingTransfers(0)]
    (
      PublishQueryOperands
      | PublishKeyReadiness
      | ReleaseValueReaders
      | ReleaseQueryReaders,
      _,
    ) => [PublishWorkgroup]
    (InvalidQueryTileExit, SplitSingleSlot | JoinedDoubleSlot) =>
      [AwaitPendingTransfers(0)]
    (AfterKeyPublication, JoinedDoubleSlot) => [StageNextPairInOtherSlot]
    (BeforeValueComputation, SplitSingleSlot) =>
      [AwaitPendingTransfers(0), PublishWorkgroup, StageNextKey]
    (AfterValueRelease, SplitSingleSlot) => [StageNextValue]
    (AfterValueRelease, JoinedDoubleSlot) => [AdvanceTransferSlot]
    _ => []
  }
}
