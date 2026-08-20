///|
const HARD_MAX_TEXT_BYTES : Int = 1048576

///|
const HARD_MAX_INPUT_TOKENS : Int = 262144

///|
const HARD_MAX_NEW_TOKENS : Int = 65536

///|
const HARD_MAX_CONTEXT_TOKENS : Int = 262144

///|
const HARD_MAX_TOKEN_ID : Int = 16777215

///|
const HARD_MAX_STOP_TOKEN_IDS : Int = 256

///|
const HARD_MAX_STOP_STRINGS : Int = 64

///|
const HARD_MAX_STOP_STRING_BYTES : Int = 4096

///|
const HARD_MAX_TRACE_BYTES : Int = 256

///|
const HARD_MAX_CACHE_SCOPE_BYTES : Int = 256

///|
const HARD_MAX_DECODED_DELTA_BYTES : Int = 16384

///|
const HARD_MAX_DEADLINE_MILLIS : UInt64 = 86400000UL

///|
const HARD_MAX_TOP_K : Int = 1048576

///|
const HARD_MAX_TEMPERATURE : Double = 100.0

///|
const PUBLIC_ERROR_CODE_BYTES : Int = 64

///|
/// Immutable resource bounds applied before the contract owns request data.
///
/// The constructor also caps deployment-provided limits at protocol hard
/// ceilings. Consequently a validated request cannot turn a permissive or
/// malformed service configuration into an unbounded allocation.
pub struct InferenceLimits {
  priv max_text_bytes : Int
  priv max_input_tokens : Int
  priv max_new_tokens : Int
  priv max_context_tokens : Int
  priv max_token_id : Int
  priv max_stop_token_ids : Int
  priv max_stop_strings : Int
  priv max_stop_string_bytes : Int
  priv max_trace_bytes : Int
  priv max_cache_scope_bytes : Int
  priv max_decoded_delta_bytes : Int
  priv max_deadline_millis : UInt64
  priv max_top_k : Int
  priv max_temperature : Double
} derive(Debug, Eq)

///|
fn positive_bounded(value : Int, maximum : Int) -> Bool {
  value > 0 && value <= maximum
}

///|
/// Constructs a complete set of v1 inference-request limits.
pub fn InferenceLimits::new(
  max_text_bytes~ : Int,
  max_input_tokens~ : Int,
  max_new_tokens~ : Int,
  max_context_tokens~ : Int,
  max_token_id~ : Int,
  max_stop_token_ids~ : Int,
  max_stop_strings~ : Int,
  max_stop_string_bytes~ : Int,
  max_trace_bytes~ : Int,
  max_cache_scope_bytes~ : Int,
  max_decoded_delta_bytes~ : Int,
  max_deadline_millis~ : UInt64,
  max_top_k~ : Int,
  max_temperature~ : Double,
) -> InferenceLimits raise InferenceContractError {
  let limits : InferenceLimits = {
    max_text_bytes,
    max_input_tokens,
    max_new_tokens,
    max_context_tokens,
    max_token_id,
    max_stop_token_ids,
    max_stop_strings,
    max_stop_string_bytes,
    max_trace_bytes,
    max_cache_scope_bytes,
    max_decoded_delta_bytes,
    max_deadline_millis,
    max_top_k,
    max_temperature,
  }
  limits.validate()
  limits
}

///|
fn InferenceLimits::validate(
  self : InferenceLimits,
) -> Unit raise InferenceContractError {
  let valid = positive_bounded(self.max_text_bytes, HARD_MAX_TEXT_BYTES) &&
    positive_bounded(self.max_input_tokens, HARD_MAX_INPUT_TOKENS) &&
    positive_bounded(self.max_new_tokens, HARD_MAX_NEW_TOKENS) &&
    positive_bounded(self.max_context_tokens, HARD_MAX_CONTEXT_TOKENS) &&
    self.max_input_tokens <= self.max_context_tokens &&
    self.max_new_tokens <= self.max_context_tokens &&
    self.max_token_id >= 0 &&
    self.max_token_id <= HARD_MAX_TOKEN_ID &&
    positive_bounded(self.max_stop_token_ids, HARD_MAX_STOP_TOKEN_IDS) &&
    positive_bounded(self.max_stop_strings, HARD_MAX_STOP_STRINGS) &&
    positive_bounded(self.max_stop_string_bytes, HARD_MAX_STOP_STRING_BYTES) &&
    positive_bounded(self.max_trace_bytes, HARD_MAX_TRACE_BYTES) &&
    positive_bounded(self.max_cache_scope_bytes, HARD_MAX_CACHE_SCOPE_BYTES) &&
    positive_bounded(self.max_decoded_delta_bytes, HARD_MAX_DECODED_DELTA_BYTES) &&
    self.max_deadline_millis > 0UL &&
    self.max_deadline_millis <= HARD_MAX_DEADLINE_MILLIS &&
    positive_bounded(self.max_top_k, HARD_MAX_TOP_K) &&
    self.max_top_k <= self.max_token_id + 1 &&
    !self.max_temperature.is_nan() &&
    !self.max_temperature.is_inf() &&
    self.max_temperature > 0.0 &&
    self.max_temperature <= HARD_MAX_TEMPERATURE
  if !valid {
    raise invalid(Limit, OutOfRange, 0)
  }
}

///|
pub fn InferenceLimits::max_text_bytes(self : InferenceLimits) -> Int {
  self.max_text_bytes
}

///|
pub fn InferenceLimits::max_input_tokens(self : InferenceLimits) -> Int {
  self.max_input_tokens
}

///|
pub fn InferenceLimits::max_new_tokens(self : InferenceLimits) -> Int {
  self.max_new_tokens
}

///|
pub fn InferenceLimits::max_context_tokens(self : InferenceLimits) -> Int {
  self.max_context_tokens
}

///|
pub fn InferenceLimits::max_token_id(self : InferenceLimits) -> Int {
  self.max_token_id
}

///|
pub fn InferenceLimits::max_stop_token_ids(self : InferenceLimits) -> Int {
  self.max_stop_token_ids
}

///|
pub fn InferenceLimits::max_stop_strings(self : InferenceLimits) -> Int {
  self.max_stop_strings
}

///|
pub fn InferenceLimits::max_stop_string_bytes(self : InferenceLimits) -> Int {
  self.max_stop_string_bytes
}

///|
pub fn InferenceLimits::max_trace_bytes(self : InferenceLimits) -> Int {
  self.max_trace_bytes
}

///|
pub fn InferenceLimits::max_cache_scope_bytes(self : InferenceLimits) -> Int {
  self.max_cache_scope_bytes
}

///|
pub fn InferenceLimits::max_decoded_delta_bytes(self : InferenceLimits) -> Int {
  self.max_decoded_delta_bytes
}

///|
pub fn InferenceLimits::max_deadline_millis(self : InferenceLimits) -> UInt64 {
  self.max_deadline_millis
}

///|
pub fn InferenceLimits::max_top_k(self : InferenceLimits) -> Int {
  self.max_top_k
}

///|
pub fn InferenceLimits::max_temperature(self : InferenceLimits) -> Double {
  self.max_temperature
}
