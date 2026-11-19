///|
const LUNA_REQUEST_SEMANTIC_IDLE : Int32 = 0

///|
const LUNA_REQUEST_SEMANTIC_WRITING : Int32 = 1

///|
const LUNA_REQUEST_SEMANTIC_TOKEN_VALUE : Int32 = 2

///|
const LUNA_REQUEST_SEMANTIC_TOKEN_DUPLICATE : Int32 = 3

///|
const LUNA_REQUEST_SEMANTIC_STRING_HEADER : Int32 = 4

///|
const LUNA_REQUEST_SEMANTIC_STRING_UTF8 : Int32 = 5

///|
const LUNA_REQUEST_SEMANTIC_STRING_DUPLICATE_LENGTH : Int32 = 6

///|
const LUNA_REQUEST_SEMANTIC_STRING_DUPLICATE_BYTE : Int32 = 7

///|
const LUNA_REQUEST_SEMANTIC_CACHE_HEADER : Int32 = 8

///|
const LUNA_REQUEST_SEMANTIC_CACHE_BYTE : Int32 = 9

///|
const LUNA_REQUEST_SEMANTIC_SORT_COPY : Int32 = 10

///|
const LUNA_REQUEST_SEMANTIC_SORT_LOAD : Int32 = 11

///|
const LUNA_REQUEST_SEMANTIC_SORT_COMPARE : Int32 = 12

///|
const LUNA_REQUEST_SEMANTIC_READY : Int32 = 13

///|
const LUNA_REQUEST_SEMANTIC_LEASED : Int32 = 14

///|
const LUNA_REQUEST_SEMANTIC_FAILED : Int32 = 15

///|
const LUNA_REQUEST_SEMANTIC_MAX_STEP_WORK_UNITS : Int32 = 65536

///|
const LUNA_REQUEST_SEMANTIC_FAILURE_STOP_TOKEN : Int32 = 0

///|
const LUNA_REQUEST_SEMANTIC_FAILURE_STOP_STRING : Int32 = 1

///|
const LUNA_REQUEST_SEMANTIC_FAILURE_CACHE_SCOPE : Int32 = 2

///|
const LUNA_REQUEST_SEMANTIC_ISSUE_EMPTY : Int32 = 0

///|
const LUNA_REQUEST_SEMANTIC_ISSUE_TOO_LONG : Int32 = 1

///|
const LUNA_REQUEST_SEMANTIC_ISSUE_INVALID_FORMAT : Int32 = 2

///|
const LUNA_REQUEST_SEMANTIC_ISSUE_OUT_OF_RANGE : Int32 = 3

///|
const LUNA_REQUEST_SEMANTIC_ISSUE_DUPLICATE : Int32 = 4

///|
#valtype
pub struct LunaRequestSemanticStepBudget {
  priv work_units : Int32
}

///|
pub fn LunaRequestSemanticStepBudget::new(
  work_units : Int32,
) -> LunaRequestSemanticStepBudget raise InferenceContractError {
  if work_units <= 0 || work_units > LUNA_REQUEST_SEMANTIC_MAX_STEP_WORK_UNITS {
    raise invalid(LunaRequestSemantic, OutOfRange, 0)
  }
  { work_units, }
}

///|
pub fn LunaRequestSemanticStepBudget::work_units(
  self : LunaRequestSemanticStepBudget,
) -> Int32 {
  self.work_units
}

///|
#valtype
pub struct LunaRequestSemanticProgress {
  priv state : Int32
}

///|
pub fn LunaRequestSemanticProgress::is_pending(
  self : LunaRequestSemanticProgress,
) -> Bool {
  self.state == 0
}

///|
pub fn LunaRequestSemanticProgress::is_ready(
  self : LunaRequestSemanticProgress,
) -> Bool {
  self.state == 1
}

///|
pub fn LunaRequestSemanticProgress::is_failed(
  self : LunaRequestSemanticProgress,
) -> Bool {
  self.state == 2
}

///|
#valtype
pub struct LunaRequestSemanticState {
  priv state : Int32
}

///|
pub fn LunaRequestSemanticState::is_validating(
  self : LunaRequestSemanticState,
) -> Bool {
  self.state == 0
}

///|
pub fn LunaRequestSemanticState::is_ready(
  self : LunaRequestSemanticState,
) -> Bool {
  self.state == 1
}

///|
pub fn LunaRequestSemanticState::is_failed(
  self : LunaRequestSemanticState,
) -> Bool {
  self.state == 2
}

///|
#valtype
pub struct LunaRequestSemanticFailure {
  priv field_code : Int32
  priv issue_code : Int32
  priv index : Int32
}

///|
pub fn LunaRequestSemanticFailure::field(
  self : LunaRequestSemanticFailure,
) -> ContractField {
  match self.field_code {
    LUNA_REQUEST_SEMANTIC_FAILURE_STOP_TOKEN => StopToken
    LUNA_REQUEST_SEMANTIC_FAILURE_STOP_STRING => StopString
    _ => CacheScope
  }
}

///|
pub fn LunaRequestSemanticFailure::issue(
  self : LunaRequestSemanticFailure,
) -> ContractIssue {
  match self.issue_code {
    LUNA_REQUEST_SEMANTIC_ISSUE_EMPTY => Empty
    LUNA_REQUEST_SEMANTIC_ISSUE_TOO_LONG => TooLong
    LUNA_REQUEST_SEMANTIC_ISSUE_INVALID_FORMAT => InvalidFormat
    LUNA_REQUEST_SEMANTIC_ISSUE_OUT_OF_RANGE => OutOfRange
    _ => Duplicate
  }
}

///|
pub fn LunaRequestSemanticFailure::index(
  self : LunaRequestSemanticFailure,
) -> Int32 {
  self.index
}

///|
#valtype
pub struct LunaRequestSemanticStopTokenStatus {
  priv state : Int32
  priv work_units : Int32
}

///|
pub fn LunaRequestSemanticStopTokenStatus::is_present(
  self : LunaRequestSemanticStopTokenStatus,
) -> Bool {
  self.state == 0
}

///|
pub fn LunaRequestSemanticStopTokenStatus::is_absent(
  self : LunaRequestSemanticStopTokenStatus,
) -> Bool {
  self.state == 1
}

///|
pub fn LunaRequestSemanticStopTokenStatus::is_stale(
  self : LunaRequestSemanticStopTokenStatus,
) -> Bool {
  self.state == 2
}

///|
/// Exact sorted-index comparisons consumed by this bounded lookup.
pub fn LunaRequestSemanticStopTokenStatus::work_units(
  self : LunaRequestSemanticStopTokenStatus,
) -> Int32 {
  self.work_units
}

///|
pub struct LunaRequestSemanticStorage {
  priv limits : InferenceLimits
  priv budget : LunaRequestSemanticStepBudget
  priv stop_tokens : FixedArray[Int32]
  priv sorted_stop_tokens : FixedArray[Int32]
  priv stop_string_offsets : FixedArray[Int32]
  priv stop_string_lengths : FixedArray[Int32]
  priv stop_string_bytes : FixedArray[Byte]
  priv cache_scope_bytes : FixedArray[Byte]
  priv mut epoch : UInt64
  priv mut retention_cursor : UInt64
  priv mut retention_count : Int32
  priv mut retention_id_a : UInt64
  priv mut retention_id_b : UInt64
  priv mut phase : Int32
  priv mut stop_token_count : Int32
  priv mut stop_string_count : Int32
  priv mut cache_scope_length : Int32
  priv mut cache_permission : CachePermission
  priv mut write_token_index : Int32
  priv mut write_string_index : Int32
  priv mut write_string_remaining : Int32
  priv mut write_stop_byte_index : Int32
  priv mut write_cache_index : Int32
  priv mut item_index : Int32
  priv mut prior_index : Int32
  priv mut byte_index : Int32
  priv mut utf8_remaining : Int32
  priv mut utf8_codepoint : Int32
  priv mut utf8_minimum : Int32
  priv mut sort_index : Int32
  priv mut sort_position : Int32
  priv mut sort_key : Int32
  priv mut failure_field : Int32
  priv mut failure_issue : Int32
  priv mut failure_index : Int32
  priv mut last_work_units : Int32
  priv mut total_work_units : UInt64
}

///|
#valtype
pub struct LunaRequestSemanticWrite {
  priv storage : LunaRequestSemanticStorage
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaRequestSemanticWork {
  priv storage : LunaRequestSemanticStorage
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaRequestSemanticLease {
  priv storage : LunaRequestSemanticStorage
  priv epoch : UInt64
}

///|
#valtype
pub struct LunaRequestSemanticView {
  priv storage : LunaRequestSemanticStorage
  priv epoch : UInt64
}

///|
/// Token-only projection of one exact semantic lease. It carries no stop
/// strings, cache data, inference limits, release authority, or raw storage.
#valtype
pub struct LunaRequestStopTokenView {
  priv storage : LunaRequestSemanticStorage
  priv epoch : UInt64
}

///|
/// Cache-only projection of one exact semantic lease. It exposes the validated
/// permission and scalar scope bytes, but no stop data, inference limits,
/// release authority, epoch, or storage representation.
#valtype
pub struct LunaRequestPrefixView {
  priv storage : LunaRequestSemanticStorage
  priv epoch : UInt64
}

///|
/// Startup-preallocated mutable owner cell for one transferred retention.
/// It avoids both a copyable retention alias and Option boxing on admission.
pub struct LunaRequestStopTokenRetentionSlot {
  priv mut storage : LunaRequestSemanticStorage?
  priv mut epoch : UInt64
  priv mut retention_id : UInt64
}

///|
fn semantic_checked_add(
  left : UInt64,
  right : UInt64,
) -> UInt64 raise InferenceContractError {
  if left > 0xffffffffffffffffUL - right {
    raise invalid(LunaRequestSemantic, OutOfRange, 0)
  }
  left + right
}

///|
fn semantic_checked_multiply(
  left : UInt64,
  right : UInt64,
) -> UInt64 raise InferenceContractError {
  if right != 0UL && left > 0xffffffffffffffffUL / right {
    raise invalid(LunaRequestSemantic, OutOfRange, 0)
  }
  left * right
}

///|
/// Exact startup `Int` cells: raw+sorted stop tokens and string offsets+lengths.
pub fn LunaRequestSemanticStorage::required_int_cells(
  limits : InferenceLimits,
) -> UInt64 raise InferenceContractError {
  limits.validate()
  semantic_checked_add(
    semantic_checked_multiply(limits.max_stop_token_ids().to_uint64(), 2UL),
    semantic_checked_multiply(limits.max_stop_strings().to_uint64(), 2UL),
  )
}

///|
/// Exact startup byte cells: all bounded stop strings plus cache scope.
pub fn LunaRequestSemanticStorage::required_byte_cells(
  limits : InferenceLimits,
) -> UInt64 raise InferenceContractError {
  limits.validate()
  semantic_checked_add(
    semantic_checked_multiply(
      limits.max_stop_strings().to_uint64(),
      limits.max_stop_string_bytes().to_uint64(),
    ),
    limits.max_cache_scope_bytes().to_uint64(),
  )
}
