///|
const LUNA_TOKEN_STORAGE_IDLE : Int = 0

///|
const LUNA_TOKEN_STORAGE_WRITING : Int = 1

///|
const LUNA_TOKEN_STORAGE_SEALED : Int = 2

///|
/// Startup-preallocated token storage. Each begin/write/seal/release cycle is
/// authenticated by an exact generation; no operation clears storage
/// proportionally or exposes its mutable fixed array.
pub struct LunaTokenBufferStorage {
  priv tokens : FixedArray[Int]
  priv max_token_id : Int
  priv mut epoch : UInt64
  priv mut phase : Int
  priv mut expected_count : Int
  priv mut written_count : Int
}

///|
#valtype
pub struct LunaTokenBufferWrite {
  priv storage : LunaTokenBufferStorage
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaTokenBufferLease {
  priv storage : LunaTokenBufferStorage
  priv epoch : UInt64
  priv token_count : Int
}

///|
/// Returns the exact number of preallocated `Int` cells required by one token
/// storage owner, before any allocation occurs.
pub fn LunaTokenBufferStorage::required_int_cells(
  capacity : Int,
) -> UInt64 raise InferenceContractError {
  if capacity <= 0 {
    raise invalid(LunaTokenStorage, OutOfRange, 0)
  }
  capacity.to_uint64()
}

///|
pub fn LunaTokenBufferStorage::new(
  capacity~ : Int,
  limits~ : InferenceLimits,
) -> LunaTokenBufferStorage raise InferenceContractError {
  if capacity <= 0 || capacity > limits.max_input_tokens() {
    raise invalid(LunaTokenStorage, OutOfRange, 0)
  }
  {
    tokens: FixedArray::make(capacity, 0),
    max_token_id: limits.max_token_id(),
    epoch: 0UL,
    phase: LUNA_TOKEN_STORAGE_IDLE,
    expected_count: 0,
    written_count: 0,
  }
}

///|
pub fn LunaTokenBufferStorage::capacity(self : LunaTokenBufferStorage) -> Int {
  self.tokens.length()
}

///|
pub fn LunaTokenBufferStorage::begin(
  self : LunaTokenBufferStorage,
  token_count~ : Int,
) -> LunaTokenBufferWrite raise InferenceContractError {
  if self.phase != LUNA_TOKEN_STORAGE_IDLE {
    raise invalid(LunaTokenStorage, LunaStorageBusy, 0)
  }
  if token_count <= 0 || token_count > self.tokens.length() {
    raise invalid(LunaTokenStorage, OutOfRange, 0)
  }
  if self.epoch == 0xffffffffffffffffUL {
    raise invalid(LunaTokenStorage, OutOfRange, 0)
  }
  self.epoch += 1UL
  self.phase = LUNA_TOKEN_STORAGE_WRITING
  self.expected_count = token_count
  self.written_count = 0
  { storage: self, epoch: self.epoch }
}

///|
fn LunaTokenBufferWrite::require_current(
  self : LunaTokenBufferWrite,
) -> LunaTokenBufferStorage raise InferenceContractError {
  if self.epoch == 0UL ||
    self.storage.epoch != self.epoch ||
    self.storage.phase != LUNA_TOKEN_STORAGE_WRITING {
    raise invalid(LunaTokenStorage, LunaStorageStale, 0)
  }
  self.storage
}

///|
/// Appends exactly one validated token. Failed validation leaves the write
/// cursor unchanged and never publishes a partial buffer.
pub fn LunaTokenBufferWrite::push(
  self : LunaTokenBufferWrite,
  token_id : Int,
) -> Unit raise InferenceContractError {
  let storage = self.require_current()
  if storage.written_count >= storage.expected_count {
    raise invalid(LunaTokenStorage, LunaStorageLifecycle, storage.written_count)
  }
  if token_id < 0 || token_id > storage.max_token_id {
    raise invalid(InputToken, OutOfRange, storage.written_count)
  }
  storage.tokens[storage.written_count] = token_id
  storage.written_count += 1
}

///|
pub fn LunaTokenBufferWrite::written_count(
  self : LunaTokenBufferWrite,
) -> Int raise InferenceContractError {
  self.require_current().written_count
}

///|
pub fn LunaTokenBufferWrite::abort(
  self : LunaTokenBufferWrite,
) -> Unit raise InferenceContractError {
  let storage = self.require_current()
  storage.phase = LUNA_TOKEN_STORAGE_IDLE
  storage.expected_count = 0
  storage.written_count = 0
}

///|
pub fn LunaTokenBufferWrite::seal(
  self : LunaTokenBufferWrite,
) -> LunaTokenBufferLease raise InferenceContractError {
  let storage = self.require_current()
  if storage.written_count != storage.expected_count {
    raise invalid(LunaTokenStorage, LunaStorageLifecycle, storage.written_count)
  }
  storage.phase = LUNA_TOKEN_STORAGE_SEALED
  { storage, epoch: self.epoch, token_count: storage.expected_count }
}

///|
fn LunaTokenBufferLease::require_current(
  self : LunaTokenBufferLease,
) -> LunaTokenBufferStorage raise InferenceContractError {
  if !self.storage.is_sealed_epoch(self.epoch, self.token_count) {
    raise invalid(LunaTokenStorage, LunaStorageStale, 0)
  }
  self.storage
}

///|
pub fn LunaTokenBufferLease::token_buffer(
  self : LunaTokenBufferLease,
) -> TokenBuffer raise InferenceContractError {
  ignore(self.require_current())
  {
    backing: LunaLeasedTokenBacking(self.storage, self.epoch),
    token_count: self.token_count,
  }
}

///|
/// Invalidates every TokenBuffer alias for this exact generation and returns
/// the owner to idle without clearing its fixed storage.
pub fn LunaTokenBufferLease::release(
  self : LunaTokenBufferLease,
) -> Unit raise InferenceContractError {
  let storage = self.require_current()
  storage.phase = LUNA_TOKEN_STORAGE_IDLE
  storage.expected_count = 0
  storage.written_count = 0
}

///|
fn LunaTokenBufferStorage::is_sealed_epoch(
  self : LunaTokenBufferStorage,
  epoch : UInt64,
  token_count : Int,
) -> Bool {
  epoch != 0UL &&
  self.epoch == epoch &&
  self.phase == LUNA_TOKEN_STORAGE_SEALED &&
  self.expected_count == token_count
}

///|
fn LunaTokenBufferStorage::read_status(
  self : LunaTokenBufferStorage,
  epoch : UInt64,
  token_count : Int,
  index : Int,
) -> LunaTokenBufferReadStatus {
  if !self.is_sealed_epoch(epoch, token_count) {
    return LunaTokenBufferReadStatus(2, 0)
  }
  LunaTokenBufferReadStatus(0, self.tokens[index])
}
