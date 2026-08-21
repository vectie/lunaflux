///|
fn LunaRequestSemanticWork::require_storage(
  self : LunaRequestSemanticWork,
) -> LunaRequestSemanticStorage raise InferenceContractError {
  let storage = self.storage
  if !storage.is_current_epoch(self.epoch) ||
    storage.phase < LUNA_REQUEST_SEMANTIC_TOKEN_VALUE ||
    storage.phase == LUNA_REQUEST_SEMANTIC_LEASED {
    raise invalid(LunaRequestSemantic, LunaStorageStale, 0)
  }
  storage
}

///|
fn LunaRequestSemanticStorage::fail_semantic(
  self : LunaRequestSemanticStorage,
  field : Int,
  issue : Int,
  index : Int,
) -> Unit {
  self.failure_field = field
  self.failure_issue = issue
  self.failure_index = index
  self.phase = LUNA_REQUEST_SEMANTIC_FAILED
}

///|
fn LunaRequestSemanticStorage::progress_token_value(self : Self) -> Unit {
  if self.item_index >= self.stop_token_count {
    self.item_index = 0
    self.phase = LUNA_REQUEST_SEMANTIC_STRING_HEADER
    return
  }
  let token_id = self.stop_tokens[self.item_index]
  if token_id < 0 || token_id > self.limits.max_token_id() {
    self.fail_semantic(
      LUNA_REQUEST_SEMANTIC_FAILURE_STOP_TOKEN,
      LUNA_REQUEST_SEMANTIC_ISSUE_OUT_OF_RANGE,
      self.item_index,
    )
    return
  }
  self.prior_index = 0
  self.phase = LUNA_REQUEST_SEMANTIC_TOKEN_DUPLICATE
}

///|
fn LunaRequestSemanticStorage::progress_token_duplicate(self : Self) -> Unit {
  if self.prior_index >= self.item_index {
    self.item_index += 1
    self.phase = LUNA_REQUEST_SEMANTIC_TOKEN_VALUE
    return
  }
  if self.stop_tokens[self.prior_index] == self.stop_tokens[self.item_index] {
    self.fail_semantic(
      LUNA_REQUEST_SEMANTIC_FAILURE_STOP_TOKEN,
      LUNA_REQUEST_SEMANTIC_ISSUE_DUPLICATE,
      self.item_index,
    )
    return
  }
  self.prior_index += 1
}

///|
fn LunaRequestSemanticStorage::progress_string_header(self : Self) -> Unit {
  if self.item_index >= self.stop_string_count {
    self.phase = LUNA_REQUEST_SEMANTIC_CACHE_HEADER
    return
  }
  let length = self.stop_string_lengths[self.item_index]
  if length == 0 {
    self.fail_semantic(
      LUNA_REQUEST_SEMANTIC_FAILURE_STOP_STRING,
      LUNA_REQUEST_SEMANTIC_ISSUE_EMPTY,
      self.item_index,
    )
    return
  }
  if length > self.limits.max_stop_string_bytes() {
    self.fail_semantic(
      LUNA_REQUEST_SEMANTIC_FAILURE_STOP_STRING,
      LUNA_REQUEST_SEMANTIC_ISSUE_TOO_LONG,
      self.item_index,
    )
    return
  }
  self.byte_index = 0
  self.utf8_remaining = 0
  self.utf8_codepoint = 0
  self.utf8_minimum = 0
  self.phase = LUNA_REQUEST_SEMANTIC_STRING_UTF8
}

///|
fn LunaRequestSemanticStorage::fail_string_format(self : Self) -> Unit {
  self.fail_semantic(
    LUNA_REQUEST_SEMANTIC_FAILURE_STOP_STRING,
    LUNA_REQUEST_SEMANTIC_ISSUE_INVALID_FORMAT,
    self.item_index,
  )
}

///|
fn LunaRequestSemanticStorage::progress_string_utf8(self : Self) -> Unit {
  let length = self.stop_string_lengths[self.item_index]
  if self.byte_index >= length {
    if self.utf8_remaining != 0 {
      self.fail_string_format()
    } else {
      self.prior_index = 0
      self.phase = LUNA_REQUEST_SEMANTIC_STRING_DUPLICATE_LENGTH
    }
    return
  }
  let offset = self.stop_string_offsets[self.item_index]
  let value = self.stop_string_bytes[offset + self.byte_index].to_int()
  self.byte_index += 1
  if self.utf8_remaining == 0 {
    if value <= 0x7f {
      return
    }
    if value >= 0xc2 && value <= 0xdf {
      self.utf8_remaining = 1
      self.utf8_codepoint = value & 0x1f
      self.utf8_minimum = 0x80
      return
    }
    if value >= 0xe0 && value <= 0xef {
      self.utf8_remaining = 2
      self.utf8_codepoint = value & 0x0f
      self.utf8_minimum = 0x800
      return
    }
    if value >= 0xf0 && value <= 0xf4 {
      self.utf8_remaining = 3
      self.utf8_codepoint = value & 0x07
      self.utf8_minimum = 0x10000
      return
    }
    self.fail_string_format()
    return
  }
  if value < 0x80 || value > 0xbf {
    self.fail_string_format()
    return
  }
  self.utf8_codepoint = (self.utf8_codepoint << 6) | (value & 0x3f)
  self.utf8_remaining -= 1
  if self.utf8_remaining == 0 &&
    (
      self.utf8_codepoint < self.utf8_minimum ||
      self.utf8_codepoint > 0x10ffff ||
      (self.utf8_codepoint >= 0xd800 && self.utf8_codepoint <= 0xdfff)
    ) {
    self.fail_string_format()
  }
}

///|
fn LunaRequestSemanticStorage::progress_string_duplicate_length(
  self : Self,
) -> Unit {
  if self.prior_index >= self.item_index {
    self.item_index += 1
    self.phase = LUNA_REQUEST_SEMANTIC_STRING_HEADER
    return
  }
  if self.stop_string_lengths[self.prior_index] !=
    self.stop_string_lengths[self.item_index] {
    self.prior_index += 1
    return
  }
  self.byte_index = 0
  self.phase = LUNA_REQUEST_SEMANTIC_STRING_DUPLICATE_BYTE
}

///|
fn LunaRequestSemanticStorage::progress_string_duplicate_byte(
  self : Self,
) -> Unit {
  let length = self.stop_string_lengths[self.item_index]
  let left = self.stop_string_offsets[self.prior_index]
  let right = self.stop_string_offsets[self.item_index]
  if self.stop_string_bytes[left + self.byte_index] !=
    self.stop_string_bytes[right + self.byte_index] {
    self.prior_index += 1
    self.phase = LUNA_REQUEST_SEMANTIC_STRING_DUPLICATE_LENGTH
    return
  }
  self.byte_index += 1
  if self.byte_index == length {
    self.fail_semantic(
      LUNA_REQUEST_SEMANTIC_FAILURE_STOP_STRING,
      LUNA_REQUEST_SEMANTIC_ISSUE_DUPLICATE,
      self.item_index,
    )
  }
}

///|
fn LunaRequestSemanticStorage::progress_cache_header(self : Self) -> Unit {
  if self.cache_scope_length == 0 {
    self.fail_semantic(
      LUNA_REQUEST_SEMANTIC_FAILURE_CACHE_SCOPE,
      LUNA_REQUEST_SEMANTIC_ISSUE_EMPTY,
      0,
    )
    return
  }
  if self.cache_scope_length > self.limits.max_cache_scope_bytes() {
    self.fail_semantic(
      LUNA_REQUEST_SEMANTIC_FAILURE_CACHE_SCOPE,
      LUNA_REQUEST_SEMANTIC_ISSUE_TOO_LONG,
      0,
    )
    return
  }
  self.byte_index = 0
  self.phase = LUNA_REQUEST_SEMANTIC_CACHE_BYTE
}

///|
fn LunaRequestSemanticStorage::progress_cache_byte(self : Self) -> Unit {
  if self.byte_index >= self.cache_scope_length {
    self.item_index = 0
    self.phase = LUNA_REQUEST_SEMANTIC_SORT_COPY
    return
  }
  let byte = self.cache_scope_bytes[self.byte_index]
  if !is_safe_identifier_byte(byte) {
    self.fail_semantic(
      LUNA_REQUEST_SEMANTIC_FAILURE_CACHE_SCOPE,
      LUNA_REQUEST_SEMANTIC_ISSUE_INVALID_FORMAT,
      0,
    )
    return
  }
  self.byte_index += 1
}

///|
fn LunaRequestSemanticStorage::progress_sort_copy(self : Self) -> Unit {
  if self.item_index >= self.stop_token_count {
    self.sort_index = 1
    self.phase = LUNA_REQUEST_SEMANTIC_SORT_LOAD
    return
  }
  self.sorted_stop_tokens[self.item_index] = self.stop_tokens[self.item_index]
  self.item_index += 1
}

///|
fn LunaRequestSemanticStorage::progress_sort_load(self : Self) -> Unit {
  if self.sort_index >= self.stop_token_count {
    self.phase = LUNA_REQUEST_SEMANTIC_READY
    return
  }
  self.sort_key = self.sorted_stop_tokens[self.sort_index]
  self.sort_position = self.sort_index - 1
  self.phase = LUNA_REQUEST_SEMANTIC_SORT_COMPARE
}

///|
fn LunaRequestSemanticStorage::progress_sort_compare(self : Self) -> Unit {
  if self.sort_position >= 0 &&
    self.sorted_stop_tokens[self.sort_position] > self.sort_key {
    self.sorted_stop_tokens[self.sort_position + 1] = self.sorted_stop_tokens[self.sort_position]
    self.sort_position -= 1
    return
  }
  self.sorted_stop_tokens[self.sort_position + 1] = self.sort_key
  self.sort_index += 1
  self.phase = LUNA_REQUEST_SEMANTIC_SORT_LOAD
}

///|
/// One call performs one documented bounded validation or index-build step.
fn LunaRequestSemanticStorage::progress_one(self : Self) -> Unit {
  match self.phase {
    LUNA_REQUEST_SEMANTIC_TOKEN_VALUE => self.progress_token_value()
    LUNA_REQUEST_SEMANTIC_TOKEN_DUPLICATE => self.progress_token_duplicate()
    LUNA_REQUEST_SEMANTIC_STRING_HEADER => self.progress_string_header()
    LUNA_REQUEST_SEMANTIC_STRING_UTF8 => self.progress_string_utf8()
    LUNA_REQUEST_SEMANTIC_STRING_DUPLICATE_LENGTH =>
      self.progress_string_duplicate_length()
    LUNA_REQUEST_SEMANTIC_STRING_DUPLICATE_BYTE =>
      self.progress_string_duplicate_byte()
    LUNA_REQUEST_SEMANTIC_CACHE_HEADER => self.progress_cache_header()
    LUNA_REQUEST_SEMANTIC_CACHE_BYTE => self.progress_cache_byte()
    LUNA_REQUEST_SEMANTIC_SORT_COPY => self.progress_sort_copy()
    LUNA_REQUEST_SEMANTIC_SORT_LOAD => self.progress_sort_load()
    LUNA_REQUEST_SEMANTIC_SORT_COMPARE => self.progress_sort_compare()
    _ => ()
  }
}

///|
pub fn LunaRequestSemanticWork::progress(
  self : LunaRequestSemanticWork,
) -> LunaRequestSemanticProgress raise InferenceContractError {
  let storage = self.require_storage()
  storage.last_work_units = 0
  if storage.phase == LUNA_REQUEST_SEMANTIC_READY {
    return { state: 1 }
  }
  if storage.phase == LUNA_REQUEST_SEMANTIC_FAILED {
    return { state: 2 }
  }
  let mut used = 0
  while used < storage.budget.work_units() &&
        storage.phase != LUNA_REQUEST_SEMANTIC_READY &&
        storage.phase != LUNA_REQUEST_SEMANTIC_FAILED {
    storage.progress_one()
    used += 1
  }
  storage.last_work_units = used
  storage.total_work_units += used.to_uint64()
  if storage.phase == LUNA_REQUEST_SEMANTIC_READY {
    { state: 1 }
  } else if storage.phase == LUNA_REQUEST_SEMANTIC_FAILED {
    { state: 2 }
  } else {
    { state: 0 }
  }
}

///|
pub fn LunaRequestSemanticWork::state(
  self : LunaRequestSemanticWork,
) -> LunaRequestSemanticState raise InferenceContractError {
  let phase = self.require_storage().phase
  if phase == LUNA_REQUEST_SEMANTIC_READY {
    { state: 1 }
  } else if phase == LUNA_REQUEST_SEMANTIC_FAILED {
    { state: 2 }
  } else {
    { state: 0 }
  }
}

///|
pub fn LunaRequestSemanticWork::last_work_units(
  self : LunaRequestSemanticWork,
) -> Int raise InferenceContractError {
  self.require_storage().last_work_units
}

///|
pub fn LunaRequestSemanticWork::total_work_units(
  self : LunaRequestSemanticWork,
) -> UInt64 raise InferenceContractError {
  self.require_storage().total_work_units
}

///|
pub fn LunaRequestSemanticWork::failure(
  self : LunaRequestSemanticWork,
) -> LunaRequestSemanticFailure raise InferenceContractError {
  let storage = self.require_storage()
  if storage.phase != LUNA_REQUEST_SEMANTIC_FAILED {
    raise invalid(LunaRequestSemantic, LunaStorageLifecycle, 0)
  }
  {
    field_code: storage.failure_field,
    issue_code: storage.failure_issue,
    index: storage.failure_index,
  }
}

///|
pub fn LunaRequestSemanticWork::take_lease(
  self : LunaRequestSemanticWork,
) -> LunaRequestSemanticLease raise InferenceContractError {
  let storage = self.require_storage()
  if storage.phase != LUNA_REQUEST_SEMANTIC_READY {
    raise invalid(LunaRequestSemantic, LunaStorageLifecycle, 0)
  }
  storage.phase = LUNA_REQUEST_SEMANTIC_LEASED
  { storage, epoch: self.epoch }
}

///|
/// Failed and ready work remain authenticated until this explicit abort.
pub fn LunaRequestSemanticWork::abort(
  self : LunaRequestSemanticWork,
) -> Unit raise InferenceContractError {
  self.require_storage().return_to_idle()
}
