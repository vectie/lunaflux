///|
/// Immutable exact row for one chunk of uncached prompt computation.
pub struct PrefillRow {
  request_id : @inference.RequestId
  request_generation : @inference.RequestGeneration
  sequence_token_start : Int
  tokens : TokenRange
  page_table : PageTableView
  capabilities : CapabilityView
  completion_slot : CompletionSlot
} derive(Debug, Eq)

///|
/// Constructs one prefill row. Table bounds are revalidated against actual
/// schedule storage when the containing plan is constructed.
pub fn PrefillRow::new(
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  sequence_token_start~ : Int,
  tokens~ : TokenRange,
  page_table~ : PageTableView,
  capabilities~ : CapabilityView,
  completion_slot~ : CompletionSlot,
  limits~ : WorkerProtocolLimits,
) -> PrefillRow raise WorkerProtocolError {
  if sequence_token_start < 0 ||
    sequence_token_start > limits.max_sequence_tokens() ||
    tokens.length() > limits.max_sequence_tokens() - sequence_token_start {
    raise InvalidRowDescriptor(kind=Prefill, rule=InvalidSequencePosition)
  }
  {
    request_id,
    request_generation,
    sequence_token_start,
    tokens,
    page_table,
    capabilities,
    completion_slot,
  }
}

///|
pub fn PrefillRow::request_id(self : PrefillRow) -> @inference.RequestId {
  self.request_id
}

///|
pub fn PrefillRow::request_generation(
  self : PrefillRow,
) -> @inference.RequestGeneration {
  self.request_generation
}

///|
pub fn PrefillRow::sequence_token_start(self : PrefillRow) -> Int {
  self.sequence_token_start
}

///|
pub fn PrefillRow::tokens(self : PrefillRow) -> TokenRange {
  self.tokens
}

///|
pub fn PrefillRow::page_table(self : PrefillRow) -> PageTableView {
  self.page_table
}

///|
pub fn PrefillRow::capabilities(self : PrefillRow) -> CapabilityView {
  self.capabilities
}

///|
pub fn PrefillRow::completion_slot(self : PrefillRow) -> CompletionSlot {
  self.completion_slot
}

///|
/// Immutable exact row for one cached autoregressive decode step.
pub struct DecodeRow {
  request_id : @inference.RequestId
  request_generation : @inference.RequestGeneration
  sequence_token_position : Int
  input_token : TokenRange
  page_table : PageTableView
  capabilities : CapabilityView
  sampling : @inference.SamplingParameters
  completion_slot : CompletionSlot
} derive(Debug, Eq)

///|
/// Constructs one decode row. Decode consumes exactly one input token.
pub fn DecodeRow::new(
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  sequence_token_position~ : Int,
  input_token~ : TokenRange,
  page_table~ : PageTableView,
  capabilities~ : CapabilityView,
  sampling~ : @inference.SamplingParameters,
  completion_slot~ : CompletionSlot,
  limits~ : WorkerProtocolLimits,
) -> DecodeRow raise WorkerProtocolError {
  if input_token.length() != 1 {
    raise InvalidRowDescriptor(kind=Decode, rule=EmptyRange)
  }
  if sequence_token_position < 0 ||
    sequence_token_position >= limits.max_sequence_tokens() {
    raise InvalidRowDescriptor(kind=Decode, rule=InvalidSequencePosition)
  }
  {
    request_id,
    request_generation,
    sequence_token_position,
    input_token,
    page_table,
    capabilities,
    sampling,
    completion_slot,
  }
}

///|
pub fn DecodeRow::request_id(self : DecodeRow) -> @inference.RequestId {
  self.request_id
}

///|
pub fn DecodeRow::request_generation(
  self : DecodeRow,
) -> @inference.RequestGeneration {
  self.request_generation
}

///|
pub fn DecodeRow::sequence_token_position(self : DecodeRow) -> Int {
  self.sequence_token_position
}

///|
pub fn DecodeRow::input_token(self : DecodeRow) -> TokenRange {
  self.input_token
}

///|
pub fn DecodeRow::page_table(self : DecodeRow) -> PageTableView {
  self.page_table
}

///|
pub fn DecodeRow::capabilities(self : DecodeRow) -> CapabilityView {
  self.capabilities
}

///|
pub fn DecodeRow::sampling(self : DecodeRow) -> @inference.SamplingParameters {
  self.sampling
}

///|
pub fn DecodeRow::completion_slot(self : DecodeRow) -> CompletionSlot {
  self.completion_slot
}
