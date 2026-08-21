///|
fn LunaRequestSemanticLease::require_storage(
  self : LunaRequestSemanticLease,
) -> LunaRequestSemanticStorage raise InferenceContractError {
  if !self.storage.is_current_epoch(self.epoch) ||
    self.storage.phase != LUNA_REQUEST_SEMANTIC_LEASED {
    raise invalid(LunaRequestSemantic, LunaStorageStale, 0)
  }
  self.storage
}

///|
fn LunaRequestSemanticView::require_storage(
  self : LunaRequestSemanticView,
) -> LunaRequestSemanticStorage raise InferenceContractError {
  if !self.storage.is_current_epoch(self.epoch) ||
    self.storage.phase != LUNA_REQUEST_SEMANTIC_LEASED {
    raise invalid(LunaRequestSemantic, LunaStorageStale, 0)
  }
  self.storage
}

///|
/// Creates a copyable read authority without transferring release ownership.
pub fn LunaRequestSemanticLease::view(
  self : LunaRequestSemanticLease,
) -> LunaRequestSemanticView raise InferenceContractError {
  ignore(self.require_storage())
  { storage: self.storage, epoch: self.epoch }
}

///|
/// Invalidates this lease and every retained View alias without clearing arrays.
pub fn LunaRequestSemanticLease::release(
  self : LunaRequestSemanticLease,
) -> Unit raise InferenceContractError {
  self.require_storage().return_to_idle()
}

///|
pub fn LunaRequestSemanticView::inference_limits(
  self : LunaRequestSemanticView,
) -> InferenceLimits raise InferenceContractError {
  self.require_storage().limits
}

///|
pub fn LunaRequestSemanticView::cache_permission(
  self : LunaRequestSemanticView,
) -> CachePermission raise InferenceContractError {
  self.require_storage().cache_permission
}

///|
pub fn LunaRequestSemanticView::stop_token_count(
  self : LunaRequestSemanticView,
) -> Int raise InferenceContractError {
  self.require_storage().stop_token_count
}

///|
pub fn LunaRequestSemanticView::stop_token_at(
  self : LunaRequestSemanticView,
  index : Int,
) -> Int raise InferenceContractError {
  let storage = self.require_storage()
  if index < 0 || index >= storage.stop_token_count {
    raise invalid(LunaRequestSemantic, LunaStorageLifecycle, index)
  }
  storage.stop_tokens[index]
}

///|
pub fn LunaRequestSemanticView::stop_string_count(
  self : LunaRequestSemanticView,
) -> Int raise InferenceContractError {
  self.require_storage().stop_string_count
}

///|
pub fn LunaRequestSemanticView::stop_string_length(
  self : LunaRequestSemanticView,
  index : Int,
) -> Int raise InferenceContractError {
  let storage = self.require_storage()
  if index < 0 || index >= storage.stop_string_count {
    raise invalid(LunaRequestSemantic, LunaStorageLifecycle, index)
  }
  storage.stop_string_lengths[index]
}

///|
pub fn LunaRequestSemanticView::stop_string_byte_at(
  self : LunaRequestSemanticView,
  string_index~ : Int,
  byte_index~ : Int,
) -> Byte raise InferenceContractError {
  let storage = self.require_storage()
  if string_index < 0 || string_index >= storage.stop_string_count {
    raise invalid(LunaRequestSemantic, LunaStorageLifecycle, string_index)
  }
  let length = storage.stop_string_lengths[string_index]
  if byte_index < 0 || byte_index >= length {
    raise invalid(LunaRequestSemantic, LunaStorageLifecycle, byte_index)
  }
  storage.stop_string_bytes[storage.stop_string_offsets[string_index] +
  byte_index]
}

///|
pub fn LunaRequestSemanticView::cache_scope_length(
  self : LunaRequestSemanticView,
) -> Int raise InferenceContractError {
  self.require_storage().cache_scope_length
}

///|
pub fn LunaRequestSemanticView::cache_scope_byte_at(
  self : LunaRequestSemanticView,
  index : Int,
) -> Byte raise InferenceContractError {
  let storage = self.require_storage()
  if index < 0 || index >= storage.cache_scope_length {
    raise invalid(LunaRequestSemantic, LunaStorageLifecycle, index)
  }
  storage.cache_scope_bytes[index]
}

///|
/// Performs an authenticated binary search over the budgetedly built sorted
/// token index. The status distinguishes invalidated authority from absence;
/// `work_units` is the exact number of token comparisons.
pub fn LunaRequestSemanticView::is_stop_token(
  self : LunaRequestSemanticView,
  token_id : Int,
) -> LunaRequestSemanticStopTokenStatus {
  let storage = self.storage
  if !storage.is_current_epoch(self.epoch) ||
    storage.phase != LUNA_REQUEST_SEMANTIC_LEASED {
    return { state: 2, work_units: 0 }
  }
  let mut low = 0
  let mut high = storage.stop_token_count
  let mut used = 0
  while low < high {
    let middle = low + (high - low) / 2
    used += 1
    let candidate = storage.sorted_stop_tokens[middle]
    if candidate == token_id {
      return { state: 0, work_units: used }
    }
    if candidate < token_id {
      low = middle + 1
    } else {
      high = middle
    }
  }
  { state: 1, work_units: used }
}
