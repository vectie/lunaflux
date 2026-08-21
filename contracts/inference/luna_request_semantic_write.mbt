///|
fn LunaRequestSemanticWrite::require_storage(
  self : LunaRequestSemanticWrite,
) -> LunaRequestSemanticStorage raise InferenceContractError {
  if !self.storage.is_current_epoch(self.epoch) ||
    self.storage.phase != LUNA_REQUEST_SEMANTIC_WRITING {
    raise invalid(LunaRequestSemantic, LunaStorageStale, 0)
  }
  self.storage
}

///|
/// Appends the next raw stop token; canonical range/duplicate checks are Work.
pub fn LunaRequestSemanticWrite::push_stop_token(
  self : LunaRequestSemanticWrite,
  token_id : Int,
) -> Unit raise InferenceContractError {
  let storage = self.require_storage()
  if storage.write_token_index >= storage.stop_token_count ||
    storage.write_string_index != 0 ||
    storage.write_stop_byte_index != 0 ||
    storage.write_cache_index != 0 {
    raise invalid(
      LunaRequestSemantic,
      LunaStorageLifecycle,
      storage.write_token_index,
    )
  }
  storage.stop_tokens[storage.write_token_index] = token_id
  storage.write_token_index += 1
}

///|
/// Starts the next stop string. Its declared bytes must immediately follow.
pub fn LunaRequestSemanticWrite::push_stop_string_length(
  self : LunaRequestSemanticWrite,
  byte_length : Int,
) -> Unit raise InferenceContractError {
  let storage = self.require_storage()
  if storage.write_token_index != storage.stop_token_count ||
    storage.write_string_remaining != 0 ||
    storage.write_string_index >= storage.stop_string_count ||
    storage.write_cache_index != 0 {
    raise invalid(
      LunaRequestSemantic,
      LunaStorageLifecycle,
      storage.write_string_index,
    )
  }
  if byte_length < 0 {
    raise invalid(
      LunaRequestSemantic,
      LunaStorageLifecycle,
      storage.write_string_index,
    )
  }
  let index = storage.write_string_index
  if byte_length > storage.limits.max_stop_string_bytes() {
    raise invalid(StopString, TooLong, index)
  }
  if byte_length >
    storage.stop_string_bytes.length() - storage.write_stop_byte_index {
    raise invalid(LunaRequestSemantic, LunaStorageLifecycle, index)
  }
  storage.stop_string_offsets[index] = storage.write_stop_byte_index
  storage.stop_string_lengths[index] = byte_length
  storage.write_string_remaining = byte_length
  if byte_length == 0 {
    storage.write_string_index += 1
  }
}

///|
pub fn LunaRequestSemanticWrite::push_stop_string_byte(
  self : LunaRequestSemanticWrite,
  byte : Byte,
) -> Unit raise InferenceContractError {
  let storage = self.require_storage()
  if storage.write_token_index != storage.stop_token_count ||
    storage.write_string_remaining <= 0 ||
    storage.write_cache_index != 0 {
    raise invalid(
      LunaRequestSemantic,
      LunaStorageLifecycle,
      storage.write_string_index,
    )
  }
  storage.stop_string_bytes[storage.write_stop_byte_index] = byte
  storage.write_stop_byte_index += 1
  storage.write_string_remaining -= 1
  if storage.write_string_remaining == 0 {
    storage.write_string_index += 1
  }
}

///|
pub fn LunaRequestSemanticWrite::push_cache_scope_byte(
  self : LunaRequestSemanticWrite,
  byte : Byte,
) -> Unit raise InferenceContractError {
  let storage = self.require_storage()
  if storage.write_token_index != storage.stop_token_count ||
    storage.write_string_index != storage.stop_string_count ||
    storage.write_string_remaining != 0 ||
    storage.write_cache_index >= storage.cache_scope_length {
    raise invalid(
      LunaRequestSemantic,
      LunaStorageLifecycle,
      storage.write_cache_index,
    )
  }
  storage.cache_scope_bytes[storage.write_cache_index] = byte
  storage.write_cache_index += 1
}

///|
/// Transfers exact-generation authority into bounded semantic validation.
pub fn LunaRequestSemanticWrite::finish(
  self : LunaRequestSemanticWrite,
) -> LunaRequestSemanticWork raise InferenceContractError {
  let storage = self.require_storage()
  if storage.write_token_index != storage.stop_token_count ||
    storage.write_string_index != storage.stop_string_count ||
    storage.write_string_remaining != 0 ||
    storage.write_cache_index != storage.cache_scope_length {
    raise invalid(LunaRequestSemantic, LunaStorageLifecycle, 0)
  }
  storage.phase = LUNA_REQUEST_SEMANTIC_TOKEN_VALUE
  storage.item_index = 0
  storage.prior_index = 0
  storage.byte_index = 0
  { storage, epoch: self.epoch }
}

///|
pub fn LunaRequestSemanticWrite::abort(
  self : LunaRequestSemanticWrite,
) -> Unit raise InferenceContractError {
  self.require_storage().return_to_idle()
}
