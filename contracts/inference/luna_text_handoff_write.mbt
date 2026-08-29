///|
fn LunaTextRequestHandoffWrite::require(
  self : Self,
) -> LunaTextRequestHandoffStorage raise InferenceContractError {
  if self.epoch == 0UL ||
    self.storage.epoch != self.epoch ||
    self.storage.phase != LUNA_TEXT_HANDOFF_WRITING ||
    self.storage.semantic_writes.length() != 1 {
    raise invalid(LunaRequestSemantic, LunaStorageStale, 0)
  }
  self.storage
}

///|
pub fn LunaTextRequestHandoffWrite::push_prompt_byte(
  self : Self,
  byte : Byte,
) -> Unit raise InferenceContractError {
  let storage = self.require()
  if storage.prompt_cursor >= storage.prompt_limit {
    raise invalid(
      LunaRequestSemantic,
      LunaStorageLifecycle,
      storage.prompt_cursor,
    )
  }
  storage.prompt[storage.prompt_cursor] = byte
  storage.prompt_cursor += 1
}

///|
pub fn LunaTextRequestHandoffWrite::push_stop_token(
  self : Self,
  token_id : Int,
) -> Unit raise InferenceContractError {
  self.require().semantic_writes[0].push_stop_token(token_id)
}

///|
pub fn LunaTextRequestHandoffWrite::push_stop_string_length(
  self : Self,
  byte_length : Int,
) -> Unit raise InferenceContractError {
  self.require().semantic_writes[0].push_stop_string_length(byte_length)
}

///|
pub fn LunaTextRequestHandoffWrite::push_stop_string_byte(
  self : Self,
  byte : Byte,
) -> Unit raise InferenceContractError {
  self.require().semantic_writes[0].push_stop_string_byte(byte)
}

///|
pub fn LunaTextRequestHandoffWrite::finish(
  self : Self,
) -> LunaTextRequestHandoffWork raise InferenceContractError {
  let storage = self.require()
  if storage.prompt_cursor <= 0 || storage.prompt_cursor > storage.prompt_limit {
    raise invalid(
      LunaRequestSemantic,
      LunaStorageLifecycle,
      storage.prompt_cursor,
    )
  }
  storage.prompt_length = storage.prompt_cursor
  for index in 0..<storage.cache_scope_length {
    storage.semantic_writes[0].push_cache_scope_byte(storage.cache_scope[index])
  }
  storage.semantic_works.push(storage.semantic_writes[0].finish())
  storage.semantic_writes.clear()
  storage.phase = LUNA_TEXT_HANDOFF_SEMANTIC
  { storage, epoch: self.epoch }
}

///|
pub fn LunaTextRequestHandoffWrite::abort(
  self : Self,
) -> Unit raise InferenceContractError {
  let storage = self.require()
  storage.semantic_writes[0].abort()
  storage.semantic_writes.clear()
  storage.reset_scalars()
  storage.phase = LUNA_TEXT_HANDOFF_IDLE
}
