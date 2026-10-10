///|
/// Startup-bound state effects. Save must retire before model submission;
/// restore must retire before host-frontier rollback is acknowledged. The
/// owner includes every persistent model/predictor allocation in its snapshot
/// and budgets the backup separately. commit discards the saved backup after
/// retirement, including poisoned ranks that can only be torn down. It is not
/// itself successful request publication. No checkpoint, CUDA or model family is
/// known to the control protocol. Teardown still belongs to the enclosing owner.
pub struct StageStateTransaction {
  priv begin_save : () -> Unit raise
  priv poll_save : () -> Bool raise
  priv commit : () -> Unit raise
  priv begin_restore : () -> Unit raise
  priv poll_restore : () -> Bool raise
}

///|
pub fn StageStateTransaction::new(
  begin_save~ : () -> Unit raise,
  poll_save~ : () -> Bool raise,
  commit~ : () -> Unit raise,
  begin_restore~ : () -> Unit raise,
  poll_restore~ : () -> Bool raise,
) -> Self {
  { begin_save, poll_save, commit, begin_restore, poll_restore, }
}
