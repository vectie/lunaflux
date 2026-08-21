///|
fn semantic_int_capacity(cells : UInt64) -> Int raise InferenceContractError {
  if cells > 2147483647UL {
    raise invalid(LunaRequestSemantic, OutOfRange, 0)
  }
  cells.to_int()
}

///|
fn semantic_stop_byte_capacity(
  limits : InferenceLimits,
) -> Int raise InferenceContractError {
  semantic_int_capacity(
    semantic_checked_multiply(
      limits.max_stop_strings().to_uint64(),
      limits.max_stop_string_bytes().to_uint64(),
    ),
  )
}

///|
pub fn LunaRequestSemanticStorage::new(
  limits : InferenceLimits,
  budget : LunaRequestSemanticStepBudget,
) -> LunaRequestSemanticStorage raise InferenceContractError {
  ignore(
    semantic_int_capacity(
      LunaRequestSemanticStorage::required_int_cells(limits),
    ),
  )
  ignore(
    semantic_int_capacity(
      LunaRequestSemanticStorage::required_byte_cells(limits),
    ),
  )
  let token_capacity = limits.max_stop_token_ids()
  let string_capacity = limits.max_stop_strings()
  {
    limits,
    budget,
    stop_tokens: FixedArray::make(token_capacity, 0),
    sorted_stop_tokens: FixedArray::make(token_capacity, 0),
    stop_string_offsets: FixedArray::make(string_capacity, 0),
    stop_string_lengths: FixedArray::make(string_capacity, 0),
    stop_string_bytes: FixedArray::make(
      semantic_stop_byte_capacity(limits),
      b'\x00',
    ),
    cache_scope_bytes: FixedArray::make(limits.max_cache_scope_bytes(), b'\x00'),
    epoch: 0UL,
    retention_cursor: 0UL,
    retention_count: 0,
    retention_id_a: 0UL,
    retention_id_b: 0UL,
    phase: LUNA_REQUEST_SEMANTIC_IDLE,
    stop_token_count: 0,
    stop_string_count: 0,
    cache_scope_length: 0,
    cache_permission: Disabled,
    write_token_index: 0,
    write_string_index: 0,
    write_string_remaining: 0,
    write_stop_byte_index: 0,
    write_cache_index: 0,
    item_index: 0,
    prior_index: 0,
    byte_index: 0,
    utf8_remaining: 0,
    utf8_codepoint: 0,
    utf8_minimum: 0,
    sort_index: 0,
    sort_position: 0,
    sort_key: 0,
    failure_field: 0,
    failure_issue: 0,
    failure_index: 0,
    last_work_units: 0,
    total_work_units: 0UL,
  }
}

///|
fn LunaRequestSemanticStorage::reset_operation(
  self : LunaRequestSemanticStorage,
) -> Unit {
  self.stop_token_count = 0
  self.stop_string_count = 0
  self.cache_scope_length = 0
  self.cache_permission = Disabled
  self.write_token_index = 0
  self.write_string_index = 0
  self.write_string_remaining = 0
  self.write_stop_byte_index = 0
  self.write_cache_index = 0
  self.item_index = 0
  self.prior_index = 0
  self.byte_index = 0
  self.utf8_remaining = 0
  self.utf8_codepoint = 0
  self.utf8_minimum = 0
  self.sort_index = 0
  self.sort_position = 0
  self.sort_key = 0
  self.failure_field = 0
  self.failure_issue = 0
  self.failure_index = 0
  self.last_work_units = 0
  self.total_work_units = 0UL
}

///|
pub fn LunaRequestSemanticStorage::begin(
  self : LunaRequestSemanticStorage,
  stop_token_count~ : Int,
  stop_string_count~ : Int,
  cache_scope_byte_count~ : Int,
  cache_permission~ : CachePermission,
) -> LunaRequestSemanticWrite raise InferenceContractError {
  if self.phase != LUNA_REQUEST_SEMANTIC_IDLE {
    raise invalid(LunaRequestSemantic, LunaStorageBusy, 0)
  }
  self.limits.validate()
  if stop_token_count < 0 {
    raise invalid(LunaRequestSemantic, LunaStorageLifecycle, 0)
  }
  if stop_token_count > self.limits.max_stop_token_ids() {
    raise invalid(StopToken, TooMany, self.limits.max_stop_token_ids())
  }
  if stop_string_count < 0 {
    raise invalid(LunaRequestSemantic, LunaStorageLifecycle, 0)
  }
  if stop_string_count > self.limits.max_stop_strings() {
    raise invalid(StopString, TooMany, self.limits.max_stop_strings())
  }
  if cache_scope_byte_count < 0 {
    raise invalid(LunaRequestSemantic, LunaStorageLifecycle, 0)
  }
  if cache_scope_byte_count > self.limits.max_cache_scope_bytes() {
    raise invalid(CacheScope, TooLong, 0)
  }
  if self.epoch == 0xffffffffffffffffUL {
    raise invalid(LunaRequestSemantic, OutOfRange, 0)
  }
  self.epoch += 1UL
  self.reset_operation()
  self.stop_token_count = stop_token_count
  self.stop_string_count = stop_string_count
  self.cache_scope_length = cache_scope_byte_count
  self.cache_permission = cache_permission
  self.phase = LUNA_REQUEST_SEMANTIC_WRITING
  { storage: self, epoch: self.epoch }
}

///|
fn LunaRequestSemanticStorage::is_current_epoch(
  self : LunaRequestSemanticStorage,
  epoch : UInt64,
) -> Bool {
  epoch != 0UL && self.epoch == epoch
}

///|
fn LunaRequestSemanticStorage::return_to_idle(
  self : LunaRequestSemanticStorage,
) -> Unit {
  if self.retention_count != 0 ||
    self.retention_id_a != 0UL ||
    self.retention_id_b != 0UL {
    abort("Luna semantic retention invariant")
  }
  self.phase = LUNA_REQUEST_SEMANTIC_IDLE
  self.stop_token_count = 0
  self.stop_string_count = 0
  self.cache_scope_length = 0
  self.write_token_index = 0
  self.write_string_index = 0
  self.write_string_remaining = 0
  self.write_stop_byte_index = 0
  self.write_cache_index = 0
  self.last_work_units = 0
}

///|
fn LunaRequestSemanticStorage::has_retention(
  self : LunaRequestSemanticStorage,
  retention_id : UInt64,
) -> Bool {
  retention_id != 0UL &&
  (self.retention_id_a == retention_id || self.retention_id_b == retention_id)
}

///|
fn LunaRequestSemanticStorage::retain_stop_tokens(
  self : LunaRequestSemanticStorage,
  epoch : UInt64,
) -> UInt64 raise InferenceContractError {
  if !self.is_current_epoch(epoch) || self.phase != LUNA_REQUEST_SEMANTIC_LEASED {
    raise invalid(LunaRequestSemantic, LunaStorageStale, 0)
  }
  if self.retention_count >= 2 {
    raise invalid(LunaRequestSemantic, LunaStorageBusy, 0)
  }
  if self.retention_cursor == 0xffffffffffffffffUL {
    raise invalid(LunaRequestSemantic, OutOfRange, 0)
  }
  self.retention_cursor += 1UL
  let retention_id = self.retention_cursor
  if self.retention_id_a == 0UL {
    self.retention_id_a = retention_id
  } else if self.retention_id_b == 0UL {
    self.retention_id_b = retention_id
  } else {
    abort("Luna semantic retention capacity invariant")
  }
  self.retention_count += 1
  retention_id
}

///|
fn LunaRequestSemanticStorage::release_stop_tokens(
  self : LunaRequestSemanticStorage,
  epoch : UInt64,
  retention_id : UInt64,
) -> Unit raise InferenceContractError {
  if !self.is_current_epoch(epoch) ||
    self.phase != LUNA_REQUEST_SEMANTIC_LEASED ||
    !self.has_retention(retention_id) {
    raise invalid(LunaRequestSemantic, LunaStorageStale, 0)
  }
  if self.retention_id_a == retention_id {
    self.retention_id_a = 0UL
  } else {
    self.retention_id_b = 0UL
  }
  self.retention_count -= 1
}
