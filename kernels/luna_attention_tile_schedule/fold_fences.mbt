///|
/// Fence placement for a query-owned fold. This describes dependencies, not
/// instructions: publication of next K can share the release of current V
/// only when their storage is disjoint and one transfer slot is reused.
pub(all) enum AttentionFoldFences {
  SeparateReadinessAndRelease
  NextKeyPublicationAtValueRelease
} derive(Debug, Eq)

///|
pub fn plan_attention_fold_fences(
  readiness : AttentionTransferReadiness,
  query_owned~ : Bool,
  transfer_slots~ : Int,
) -> AttentionFoldFences {
  if query_owned &&
    transfer_slots == 1 &&
    readiness == KeyScoreBeforeValueReadiness {
    NextKeyPublicationAtValueRelease
  } else {
    SeparateReadinessAndRelease
  }
}
