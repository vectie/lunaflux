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
/// Issues a scheduler-safe token-only projection of this exact lease epoch.
pub fn LunaRequestSemanticLease::stop_token_view(
  self : LunaRequestSemanticLease,
) -> LunaRequestStopTokenView raise InferenceContractError {
  ignore(self.require_storage())
  { storage: self.storage, epoch: self.epoch }
}

///|
/// Issues the narrow cache-policy projection used by the scheduler prefix
/// owner. The same semantic lease epoch authenticates every later scalar read.
pub fn LunaRequestSemanticLease::prefix_view(
  self : LunaRequestSemanticLease,
) -> LunaRequestPrefixView raise InferenceContractError {
  ignore(self.require_storage())
  { storage: self.storage, epoch: self.epoch }
}

///|
/// Reports whether this exact cache-policy projection remains live.
pub fn LunaRequestPrefixView::is_live(self : LunaRequestPrefixView) -> Bool {
  self.storage.is_current_epoch(self.epoch) &&
  self.storage.phase == LUNA_REQUEST_SEMANTIC_LEASED
}

///|
/// Proves that the scheduler's cache and stop-token projections came from the
/// same exact live semantic lease. It exposes no owner or epoch scalar.
pub fn LunaRequestPrefixView::binds_stop_tokens(
  self : LunaRequestPrefixView,
  stop_tokens : LunaRequestStopTokenView,
) -> Bool {
  self.is_live() &&
  stop_tokens.is_live() &&
  physical_equal(self.storage, stop_tokens.storage) &&
  self.epoch == stop_tokens.epoch
}

///|
/// Returns only the validated scalar cache permission.
pub fn LunaRequestPrefixView::cache_permission(
  self : LunaRequestPrefixView,
) -> CachePermission raise InferenceContractError {
  if !self.is_live() {
    raise invalid(LunaRequestSemantic, LunaStorageStale, 0)
  }
  self.storage.cache_permission
}

///|
/// Returns the validated ASCII scope length after authenticating the epoch.
pub fn LunaRequestPrefixView::scope_length(
  self : LunaRequestPrefixView,
) -> Int raise InferenceContractError {
  if !self.is_live() {
    raise invalid(LunaRequestSemantic, LunaStorageStale, 0)
  }
  self.storage.cache_scope_length
}

///|
/// Reads one validated ASCII scope byte. Stale authentication precedes bounds.
pub fn LunaRequestPrefixView::scope_byte_at(
  self : LunaRequestPrefixView,
  index : Int,
) -> Byte raise InferenceContractError {
  if !self.is_live() {
    raise invalid(LunaRequestSemantic, LunaStorageStale, 0)
  }
  if index < 0 || index >= self.storage.cache_scope_length {
    raise invalid(LunaRequestSemantic, LunaStorageLifecycle, index)
  }
  self.storage.cache_scope_bytes[index]
}

///|
/// Invalidates the lease and Views only after both lower retention slots release.
pub fn LunaRequestSemanticLease::release(
  self : LunaRequestSemanticLease,
) -> Unit raise InferenceContractError {
  let storage = self.require_storage()
  if storage.retention_count != 0 {
    raise invalid(LunaRequestSemantic, LunaStorageBusy, 0)
  }
  storage.return_to_idle()
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
fn LunaRequestSemanticStorage::stop_token_status(
  self : LunaRequestSemanticStorage,
  epoch : UInt64,
  token_id : Int,
) -> LunaRequestSemanticStopTokenStatus {
  if !self.is_current_epoch(epoch) || self.phase != LUNA_REQUEST_SEMANTIC_LEASED {
    return { state: 2, work_units: 0 }
  }
  let mut low = 0
  let mut high = self.stop_token_count
  let mut used = 0
  while low < high {
    let middle = low + (high - low) / 2
    used += 1
    let candidate = self.sorted_stop_tokens[middle]
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

///|
pub fn LunaRequestSemanticView::is_stop_token(
  self : LunaRequestSemanticView,
  token_id : Int,
) -> LunaRequestSemanticStopTokenStatus {
  self.storage.stop_token_status(self.epoch, token_id)
}

///|
/// Reports whether this exact token-only lease projection remains live.
pub fn LunaRequestStopTokenView::is_live(
  self : LunaRequestStopTokenView,
) -> Bool {
  self.storage.is_current_epoch(self.epoch) &&
  self.storage.phase == LUNA_REQUEST_SEMANTIC_LEASED
}

///|
/// Bounded authenticated membership without exposing token storage.
pub fn LunaRequestStopTokenView::is_stop_token(
  self : LunaRequestStopTokenView,
  token_id : Int,
) -> LunaRequestSemanticStopTokenStatus {
  self.storage.stop_token_status(self.epoch, token_id)
}

///|
/// Acquires one of the two exact retentions directly into a caller-owned,
/// startup-preallocated empty slot.
pub fn LunaRequestStopTokenView::retain_into(
  self : LunaRequestStopTokenView,
  slot : LunaRequestStopTokenRetentionSlot,
) -> Unit raise InferenceContractError {
  slot.bind(self.storage, self.epoch)
}

///|
/// Allocates one empty owner cell during enclosing-owner startup.
pub fn LunaRequestStopTokenRetentionSlot::new() -> LunaRequestStopTokenRetentionSlot {
  { storage: None, epoch: 0UL, retention_id: 0UL }
}

///|
/// Transfers one exact retention into this already allocated empty cell.
fn LunaRequestStopTokenRetentionSlot::bind(
  self : LunaRequestStopTokenRetentionSlot,
  storage : LunaRequestSemanticStorage,
  epoch : UInt64,
) -> Unit raise InferenceContractError {
  match self.storage {
    Some(_) => raise invalid(LunaRequestSemantic, LunaStorageBusy, 0)
    None => ()
  }
  let retention_id = storage.retain_stop_tokens(epoch)
  self.storage = Some(storage)
  self.epoch = epoch
  self.retention_id = retention_id
}

///|
/// Reports whether the transferred exact retention remains live.
pub fn LunaRequestStopTokenRetentionSlot::is_live(
  self : LunaRequestStopTokenRetentionSlot,
) -> Bool {
  match self.storage {
    Some(storage) =>
      storage.is_current_epoch(self.epoch) &&
      storage.phase == LUNA_REQUEST_SEMANTIC_LEASED &&
      storage.has_retention(self.retention_id)
    None => false
  }
}

///|
/// Bounded membership through this transferred owner cell.
pub fn LunaRequestStopTokenRetentionSlot::is_stop_token(
  self : LunaRequestStopTokenRetentionSlot,
  token_id : Int,
) -> LunaRequestSemanticStopTokenStatus {
  match self.storage {
    Some(storage) if self.is_live() =>
      storage.stop_token_status(self.epoch, token_id)
    _ => { state: 2, work_units: 0 }
  }
}

///|
/// Releases the exact bound retention, leaving this cell reusable.
pub fn LunaRequestStopTokenRetentionSlot::release(
  self : LunaRequestStopTokenRetentionSlot,
) -> Unit raise InferenceContractError {
  let storage = match self.storage {
    Some(value) => value
    None => raise invalid(LunaRequestSemantic, LunaStorageStale, 0)
  }
  storage.release_stop_tokens(self.epoch, self.retention_id)
  self.storage = None
  self.epoch = 0UL
  self.retention_id = 0UL
}
