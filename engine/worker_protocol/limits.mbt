///|
const HARD_MAX_ROWS : Int = 4096

///|
const HARD_MAX_PLAN_TOKENS : Int = 1048576

///|
const HARD_MAX_PLAN_PAGES : Int = 1048576

///|
const HARD_MAX_CAPABILITIES : Int = 16384

///|
const HARD_MAX_SEQUENCE_TOKENS : Int = 1048576

///|
const HARD_MAX_TOKEN_ID : Int = 16777215

///|
/// Startup-owned bounds for every collection and scalar accepted by this
/// protocol package.
pub struct WorkerProtocolLimits {
  max_prefill_rows : Int
  max_decode_rows : Int
  max_plan_rows : Int
  max_plan_tokens : Int
  max_plan_pages : Int
  max_capabilities : Int
  max_completion_slots : Int
  max_sequence_tokens : Int
  max_token_id : Int
} derive(Debug, Eq)

///|
fn positive_at_most(value : Int, maximum : Int) -> Bool {
  value > 0 && value <= maximum
}

///|
/// Constructs an exact bounded worker-protocol envelope.
pub fn WorkerProtocolLimits::new(
  max_prefill_rows~ : Int,
  max_decode_rows~ : Int,
  max_plan_rows~ : Int,
  max_plan_tokens~ : Int,
  max_plan_pages~ : Int,
  max_capabilities~ : Int,
  max_completion_slots~ : Int,
  max_sequence_tokens~ : Int,
  max_token_id~ : Int,
) -> WorkerProtocolLimits raise WorkerProtocolError {
  if !positive_at_most(max_prefill_rows, HARD_MAX_ROWS) {
    raise InvalidLimit(PrefillRows)
  }
  if !positive_at_most(max_decode_rows, HARD_MAX_ROWS) {
    raise InvalidLimit(DecodeRows)
  }
  if !positive_at_most(max_plan_rows, HARD_MAX_ROWS) {
    raise InvalidLimit(PlanRows)
  }
  if max_prefill_rows > max_plan_rows || max_decode_rows > max_plan_rows {
    raise InvalidLimit(PlanRows)
  }
  if !positive_at_most(max_plan_tokens, HARD_MAX_PLAN_TOKENS) {
    raise InvalidLimit(PlanTokens)
  }
  if !positive_at_most(max_plan_pages, HARD_MAX_PLAN_PAGES) {
    raise InvalidLimit(PlanPages)
  }
  if !positive_at_most(max_capabilities, HARD_MAX_CAPABILITIES) {
    raise InvalidLimit(Capabilities)
  }
  if !positive_at_most(max_completion_slots, HARD_MAX_ROWS) ||
    max_completion_slots < max_plan_rows {
    raise InvalidLimit(CompletionSlots)
  }
  if !positive_at_most(max_sequence_tokens, HARD_MAX_SEQUENCE_TOKENS) {
    raise InvalidLimit(SequenceTokens)
  }
  if max_token_id < 0 || max_token_id > HARD_MAX_TOKEN_ID {
    raise InvalidLimit(TokenId)
  }
  {
    max_prefill_rows,
    max_decode_rows,
    max_plan_rows,
    max_plan_tokens,
    max_plan_pages,
    max_capabilities,
    max_completion_slots,
    max_sequence_tokens,
    max_token_id,
  }
}

///|
pub fn WorkerProtocolLimits::max_prefill_rows(
  self : WorkerProtocolLimits,
) -> Int {
  self.max_prefill_rows
}

///|
pub fn WorkerProtocolLimits::max_decode_rows(
  self : WorkerProtocolLimits,
) -> Int {
  self.max_decode_rows
}

///|
pub fn WorkerProtocolLimits::max_plan_rows(self : WorkerProtocolLimits) -> Int {
  self.max_plan_rows
}

///|
pub fn WorkerProtocolLimits::max_plan_tokens(
  self : WorkerProtocolLimits,
) -> Int {
  self.max_plan_tokens
}

///|
pub fn WorkerProtocolLimits::max_plan_pages(self : WorkerProtocolLimits) -> Int {
  self.max_plan_pages
}

///|
pub fn WorkerProtocolLimits::max_capabilities(
  self : WorkerProtocolLimits,
) -> Int {
  self.max_capabilities
}

///|
pub fn WorkerProtocolLimits::max_completion_slots(
  self : WorkerProtocolLimits,
) -> Int {
  self.max_completion_slots
}

///|
pub fn WorkerProtocolLimits::max_sequence_tokens(
  self : WorkerProtocolLimits,
) -> Int {
  self.max_sequence_tokens
}

///|
pub fn WorkerProtocolLimits::max_token_id(self : WorkerProtocolLimits) -> Int {
  self.max_token_id
}
