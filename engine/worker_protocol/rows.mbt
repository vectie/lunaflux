///|
/// Immutable exact row for one chunk of uncached prompt computation.
pub struct PrefillRow {
  request_id : @inference.RequestId
  request_generation : @inference.RequestGeneration
  sequence_token_start : Int
  tokens : TokenRange
  page_table : PageTableView
  capabilities : CapabilityView
  produces_token : Bool
  sampling : @inference.SamplingParameters
  sampling_seed : @inference.SamplingSeed
  sample_index : Int
  completion_slot : CompletionSlot
} derive(Eq)

///|
/// Constructs an intermediate prefill row. Intermediate chunks never sample
/// or produce an output token; the stored greedy value is an inert canonical
/// placeholder. Use `final_prompt` for the final prompt chunk.
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
    produces_token: false,
    sampling: @inference.SamplingParameters::greedy(),
    sampling_seed: try! @inference.SamplingSeed::new(1UL),
    sample_index: 0,
    completion_slot,
  }
}

///|
/// Explicit name for `new`; constructs a chunk that only advances prompt KV.
pub fn PrefillRow::intermediate(
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  sequence_token_start~ : Int,
  tokens~ : TokenRange,
  page_table~ : PageTableView,
  capabilities~ : CapabilityView,
  completion_slot~ : CompletionSlot,
  limits~ : WorkerProtocolLimits,
) -> PrefillRow raise WorkerProtocolError {
  PrefillRow::new(
    request_id~,
    request_generation~,
    sequence_token_start~,
    tokens~,
    page_table~,
    capabilities~,
    completion_slot~,
    limits~,
  )
}

///|
/// Constructs the final prompt chunk. Its worker completion must contain both
/// the exact processed-token count and the first sampled output token.
pub fn PrefillRow::final_prompt(
  request_id~ : @inference.RequestId,
  request_generation~ : @inference.RequestGeneration,
  sequence_token_start~ : Int,
  tokens~ : TokenRange,
  page_table~ : PageTableView,
  capabilities~ : CapabilityView,
  sampling~ : @inference.SamplingParameters,
  sampling_seed~ : @inference.SamplingSeed,
  sample_index~ : Int,
  completion_slot~ : CompletionSlot,
  limits~ : WorkerProtocolLimits,
) -> PrefillRow raise WorkerProtocolError {
  let row = PrefillRow::new(
    request_id~,
    request_generation~,
    sequence_token_start~,
    tokens~,
    page_table~,
    capabilities~,
    completion_slot~,
    limits~,
  )
  if sample_index < 0 || sample_index >= limits.max_sequence_tokens() {
    raise InvalidRowDescriptor(kind=Prefill, rule=InvalidSampleIndex)
  }
  { ..row, produces_token: true, sampling, sampling_seed, sample_index }
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
/// True only for the final prompt chunk, which samples exactly one token.
pub fn PrefillRow::produces_token(self : PrefillRow) -> Bool {
  self.produces_token
}

///|
/// Sampling parameters used only when `produces_token` is true.
pub fn PrefillRow::sampling(self : PrefillRow) -> @inference.SamplingParameters {
  self.sampling
}

///|
pub fn PrefillRow::sampling_seed(self : PrefillRow) -> @inference.SamplingSeed {
  self.sampling_seed
}

///|
pub fn PrefillRow::sample_index(self : PrefillRow) -> Int {
  self.sample_index
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
  sampling_seed : @inference.SamplingSeed
  sample_index : Int
  completion_slot : CompletionSlot
} derive(Eq)

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
  sampling_seed~ : @inference.SamplingSeed,
  sample_index~ : Int,
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
  if sample_index < 0 || sample_index >= limits.max_sequence_tokens() {
    raise InvalidRowDescriptor(kind=Decode, rule=InvalidSampleIndex)
  }
  {
    request_id,
    request_generation,
    sequence_token_position,
    input_token,
    page_table,
    capabilities,
    sampling,
    sampling_seed,
    sample_index,
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
pub fn DecodeRow::sampling_seed(self : DecodeRow) -> @inference.SamplingSeed {
  self.sampling_seed
}

///|
pub fn DecodeRow::sample_index(self : DecodeRow) -> Int {
  self.sample_index
}

///|
pub fn DecodeRow::completion_slot(self : DecodeRow) -> CompletionSlot {
  self.completion_slot
}
