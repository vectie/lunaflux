///|
/// Captures the current logical end of an in-progress multirow build.
/// Checkpointing is valid only while writable and with no open row draft. It
/// stores only scalar counts and allocates nothing.
pub fn SchedulePlanBuffer::checkpoint(
  self : SchedulePlanBuffer,
) -> PlanBuildCheckpoint raise WorkerProtocolError {
  self.require_state(Writable)
  if self.draft_open {
    raise PlanRowDraftAlreadyOpen
  }
  if self.checkpoint_epoch == 0xffffffffffffffffUL {
    raise IdentityOverflow(PlanBuildCheckpointEpochIdentity)
  }
  self.checkpoint_epoch = self.checkpoint_epoch + 1UL
  self.checkpoint_token_count = self.token_count
  self.checkpoint_page_count = self.page_count
  self.checkpoint_capability_count = self.capability_count
  self.checkpoint_prefill_count = self.prefill_count
  self.checkpoint_decode_count = self.decode_count
  PlanBuildCheckpoint(self, self.epoch, self.checkpoint_epoch)
}

///|
fn PlanBuildCheckpoint::authenticate_owner(
  self : PlanBuildCheckpoint,
  owner : SchedulePlanBuffer,
) -> Unit raise WorkerProtocolError {
  if !physical_equal(self.0, owner) {
    raise PlanBuildCheckpointMismatch
  }
  if self.1 != owner.epoch {
    raise StalePlanBufferEpoch
  }
  owner.require_state(Writable)
  if self.2 != owner.checkpoint_epoch {
    raise StalePlanBuildCheckpointEpoch
  }
  if owner.draft_open {
    raise PlanRowDraftAlreadyOpen
  }
}

///|
/// Restores all logical table and row ends captured by the latest
/// authenticated checkpoint. Page cells in the discarded suffix are cleared
/// before their logical count is reduced. The checkpoint is single-use; a
/// successful rollback advances its identity so repeated or superseded
/// rollback attempts fail closed.
pub fn SchedulePlanBuffer::rollback_to(
  self : SchedulePlanBuffer,
  checkpoint : PlanBuildCheckpoint,
) -> Unit raise WorkerProtocolError {
  checkpoint.authenticate_owner(self)
  if self.checkpoint_epoch == 0xffffffffffffffffUL {
    raise IdentityOverflow(PlanBuildCheckpointEpochIdentity)
  }
  if self.checkpoint_token_count > self.token_count ||
    self.checkpoint_page_count > self.page_count ||
    self.checkpoint_capability_count > self.capability_count ||
    self.checkpoint_prefill_count > self.prefill_count ||
    self.checkpoint_decode_count > self.decode_count {
    raise PlanBuildCheckpointMismatch
  }
  for index in self.checkpoint_page_count..<self.page_count {
    map_page_storage_clear(self.pages, index)
  }
  self.token_count = self.checkpoint_token_count
  self.page_count = self.checkpoint_page_count
  self.capability_count = self.checkpoint_capability_count
  self.prefill_count = self.checkpoint_prefill_count
  self.decode_count = self.checkpoint_decode_count
  self.checkpoint_epoch = self.checkpoint_epoch + 1UL
}
