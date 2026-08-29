///|
fn LunaTextRequestHandoffWork::require(
  self : Self,
) -> LunaTextRequestHandoffStorage raise InferenceContractError {
  if self.epoch == 0UL ||
    self.storage.epoch != self.epoch ||
    (
      self.storage.phase != LUNA_TEXT_HANDOFF_SEMANTIC &&
      self.storage.phase != LUNA_TEXT_HANDOFF_READY
    ) {
    raise invalid(LunaRequestSemantic, LunaStorageStale, 0)
  }
  self.storage
}

///|
pub fn LunaTextRequestHandoffWork::progress(
  self : Self,
) -> LunaRequestSemanticProgress raise InferenceContractError {
  let storage = self.require()
  if storage.phase == LUNA_TEXT_HANDOFF_READY {
    return { state: 1 }
  }
  let progress = storage.semantic_works[0].progress()
  if progress.is_failed() {
    return progress
  }
  if progress.is_ready() {
    storage.semantic_leases.push(storage.semantic_works[0].take_lease())
    storage.semantic_works.clear()
    storage.phase = LUNA_TEXT_HANDOFF_READY
    return { state: 1 }
  }
  { state: 0 }
}

///|
pub fn LunaTextRequestHandoffWork::take_lease(
  self : Self,
) -> LunaTextRequestHandoffLease raise InferenceContractError {
  let storage = self.require()
  if storage.phase != LUNA_TEXT_HANDOFF_READY ||
    storage.semantic_leases.length() != 1 {
    raise invalid(LunaRequestSemantic, LunaStorageLifecycle, 0)
  }
  storage.phase = LUNA_TEXT_HANDOFF_LEASED
  { storage, epoch: self.epoch }
}

///|
pub fn LunaTextRequestHandoffWork::abort(
  self : Self,
) -> Unit raise InferenceContractError {
  let storage = self.require()
  if storage.semantic_works.length() == 1 {
    storage.semantic_works[0].abort()
    storage.semantic_works.clear()
  }
  if storage.semantic_leases.length() == 1 {
    storage.semantic_leases[0].release()
    storage.semantic_leases.clear()
  }
  storage.reset_scalars()
  storage.phase = LUNA_TEXT_HANDOFF_IDLE
}
