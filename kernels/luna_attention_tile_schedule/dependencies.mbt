///|
/// Readiness is a dependency property, not a vendor copy instruction. In the
/// split form, QK may execute after K arrives while V remains in flight. The
/// ordered softmax/value fold starts only after V arrives. Storage in a ring
/// slot cannot be reused until that fold's last read.
pub(all) enum AttentionTransferReadiness {
  JoinedKeyValueReadiness
  KeyScoreBeforeValueReadiness
} derive(Debug, Eq)

///|
/// Pure schedule refinement. The query is an invariant of the outer row/head
/// map. Splitting transfer readiness needs both an independent score value and
/// an alternate storage slot; other compute families retain their joined fold.
pub fn AttentionTileSchedule::transfer_readiness(
  self : AttentionTileSchedule,
) -> AttentionTransferReadiness {
  if self.compute == AttentionGroupedQuerySplitSubgroupDot &&
    self.memory == AttentionAsyncStagedShared &&
    self.pipeline_stages == 2 {
    KeyScoreBeforeValueReadiness
  } else {
    JoinedKeyValueReadiness
  }
}
