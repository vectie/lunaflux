///|
/// Limits made effective by admission for one accepted request.
pub struct EffectiveLimits {
  max_new_tokens : Int
  context_ceiling : Int
} derive(Debug, Eq)

///|
pub fn EffectiveLimits::new(
  max_new_tokens~ : Int,
  context_ceiling~ : Int,
  limits : InferenceLimits,
) -> EffectiveLimits raise InferenceContractError {
  if max_new_tokens <= 0 || max_new_tokens > limits.max_new_tokens() {
    raise invalid(MaximumNewTokens, OutOfRange, 0)
  }
  if context_ceiling <= 0 || context_ceiling > limits.max_context_tokens() {
    raise invalid(ContextCeiling, OutOfRange, 0)
  }
  if max_new_tokens > context_ceiling {
    raise invalid(ContextCeiling, Inconsistent, 0)
  }
  { max_new_tokens, context_ceiling }
}

///|
pub fn EffectiveLimits::max_new_tokens(self : EffectiveLimits) -> Int {
  self.max_new_tokens
}

///|
pub fn EffectiveLimits::context_ceiling(self : EffectiveLimits) -> Int {
  self.context_ceiling
}

///|
/// Accepted stream event payload.
pub struct AcceptedEvent {
  model_identity : @spec.ModelIdentity
  effective_limits : EffectiveLimits
} derive(Debug, Eq)

///|
pub fn AcceptedEvent::new(
  model_identity~ : @spec.ModelIdentity,
  effective_limits~ : EffectiveLimits,
) -> AcceptedEvent {
  { model_identity, effective_limits }
}

///|
pub fn AcceptedEvent::model_identity(
  self : AcceptedEvent,
) -> @spec.ModelIdentity {
  self.model_identity
}

///|
pub fn AcceptedEvent::effective_limits(self : AcceptedEvent) -> EffectiveLimits {
  self.effective_limits
}

///|
/// Optional decoded token delta. It deliberately has no `Debug`.
pub struct DecodedDelta {
  text : String
  utf8_bytes : Int
} derive(Eq)

///|
pub fn DecodedDelta::from_utf8(
  bytes : BytesView,
  limits : InferenceLimits,
) -> DecodedDelta raise InferenceContractError {
  if bytes.length() == 0 {
    raise invalid(DecodedDelta, Empty, 0)
  }
  if bytes.length() > limits.max_decoded_delta_bytes() {
    raise invalid(DecodedDelta, TooLong, 0)
  }
  let text = @utf8.decode(bytes) catch {
    _ => raise invalid(DecodedDelta, InvalidFormat, 0)
  }
  { text, utf8_bytes: bytes.length() }
}

///|
pub fn DecodedDelta::text(self : DecodedDelta) -> String {
  self.text
}

///|
pub fn DecodedDelta::utf8_bytes(self : DecodedDelta) -> Int {
  self.utf8_bytes
}

///|
/// One generated token at its zero-based output position.
pub struct TokenEvent {
  token_id : Int
  decoded_delta : DecodedDelta?
  position : Int
}

///|
pub fn TokenEvent::new(
  token_id~ : Int,
  decoded_delta~ : DecodedDelta?,
  position~ : Int,
  limits : InferenceLimits,
) -> TokenEvent raise InferenceContractError {
  if token_id < 0 || token_id > limits.max_token_id() {
    raise invalid(InputToken, OutOfRange, 0)
  }
  if position < 0 || position >= limits.max_new_tokens() {
    raise invalid(TokenPosition, OutOfRange, 0)
  }
  match decoded_delta {
    Some(delta) if delta.utf8_bytes() > limits.max_decoded_delta_bytes() =>
      raise invalid(DecodedDelta, TooLong, 0)
    _ => ()
  }
  { token_id, decoded_delta, position }
}

///|
pub fn TokenEvent::token_id(self : TokenEvent) -> Int {
  self.token_id
}

///|
pub fn TokenEvent::decoded_delta(self : TokenEvent) -> DecodedDelta? {
  self.decoded_delta
}

///|
pub fn TokenEvent::position(self : TokenEvent) -> Int {
  self.position
}

///|
/// Monotonic token accounting snapshot.
pub struct Usage {
  input_tokens : Int
  cached_input_tokens : Int
  output_tokens : Int
  total_tokens : Int
} derive(Debug, Eq)

///|
/// Constructs the first usage snapshot before any output token is published.
pub fn Usage::initial(
  input_tokens~ : Int,
  cached_input_tokens~ : Int,
  limits : InferenceLimits,
) -> Usage raise InferenceContractError {
  if input_tokens < 0 || input_tokens > limits.max_context_tokens() {
    raise invalid(UsageInput, OutOfRange, 0)
  }
  if cached_input_tokens < 0 || cached_input_tokens > input_tokens {
    raise invalid(UsageCachedInput, Inconsistent, 0)
  }
  {
    input_tokens,
    cached_input_tokens,
    output_tokens: 0,
    total_tokens: input_tokens,
  }
}

///|
/// Advances only the output count; input and cached-input accounting is fixed.
pub fn Usage::advance(
  self : Usage,
  output_tokens~ : Int,
  limits : InferenceLimits,
) -> Usage raise InferenceContractError {
  if output_tokens < self.output_tokens {
    raise invalid(UsageOutput, NotMonotonic, 0)
  }
  if output_tokens > limits.max_new_tokens() {
    raise invalid(UsageOutput, OutOfRange, 0)
  }
  if self.input_tokens > limits.max_context_tokens() - output_tokens {
    raise invalid(UsageOutput, Inconsistent, 0)
  }
  {
    input_tokens: self.input_tokens,
    cached_input_tokens: self.cached_input_tokens,
    output_tokens,
    total_tokens: self.input_tokens + output_tokens,
  }
}

///|
pub fn Usage::input_tokens(self : Usage) -> Int {
  self.input_tokens
}

///|
pub fn Usage::cached_input_tokens(self : Usage) -> Int {
  self.cached_input_tokens
}

///|
pub fn Usage::output_tokens(self : Usage) -> Int {
  self.output_tokens
}

///|
pub fn Usage::total_tokens(self : Usage) -> Int {
  self.total_tokens
}

///|
/// Terminal successful outcome vocabulary.
pub(all) enum FinishReason {
  StopSequence
  TokenLimit
  Cancelled
} derive(Debug, Eq)

///|
/// Completed stream event payload. It deliberately has no `Debug` because the
/// optional final decoded tail may contain request output.
pub struct CompletedEvent {
  finish_reason : FinishReason
  final_usage : Usage
  final_delta : DecodedDelta?
} derive(Eq)

///|
pub fn CompletedEvent::new(
  finish_reason~ : FinishReason,
  final_usage~ : Usage,
) -> CompletedEvent {
  { finish_reason, final_usage, final_delta: None }
}

///|
/// Constructs a terminal event with an exact bounded final decoded tail.
pub fn CompletedEvent::with_final_delta(
  finish_reason~ : FinishReason,
  final_usage~ : Usage,
  final_delta~ : DecodedDelta,
  limits : InferenceLimits,
) -> CompletedEvent raise InferenceContractError {
  if final_delta.utf8_bytes() > limits.max_decoded_delta_bytes() {
    raise invalid(DecodedDelta, TooLong, 0)
  }
  { finish_reason, final_usage, final_delta: Some(final_delta) }
}

///|
pub fn CompletedEvent::finish_reason(self : CompletedEvent) -> FinishReason {
  self.finish_reason
}

///|
pub fn CompletedEvent::final_usage(self : CompletedEvent) -> Usage {
  self.final_usage
}

///|
/// Optional final decoded bytes flushed from an unmatched stop-prefix after
/// the last generated token. They belong to the terminal event so the stream
/// never invents a text-only token.
pub fn CompletedEvent::final_delta(self : CompletedEvent) -> DecodedDelta? {
  self.final_delta
}

///|
/// Bounded machine-readable public failure code. It deliberately has no
/// `Debug`, and its grammar cannot contain paths, whitespace, or prose.
pub struct PublicErrorCode {
  value : String
}

///|
pub fn PublicErrorCode::from_ascii(
  value : BytesView,
) -> PublicErrorCode raise InferenceContractError {
  { value: bounded_identifier(value, PUBLIC_ERROR_CODE_BYTES, PublicErrorCode) }
}

///|
pub fn PublicErrorCode::as_string(self : PublicErrorCode) -> String {
  self.value
}

///|
/// Failed stream event payload. It deliberately has no `Debug`.
pub struct FailedEvent {
  code : PublicErrorCode
  retryable : Bool
}

///|
pub fn FailedEvent::new(
  code~ : PublicErrorCode,
  retryable~ : Bool,
) -> FailedEvent {
  { code, retryable }
}

///|
pub fn FailedEvent::code(self : FailedEvent) -> PublicErrorCode {
  self.code
}

///|
pub fn FailedEvent::retryable(self : FailedEvent) -> Bool {
  self.retryable
}

///|
/// Canonical native streaming event. Payload-bearing variants deliberately do
/// not provide a derived `Debug` representation.
pub(all) enum StreamEvent {
  Accepted(AcceptedEvent)
  Token(TokenEvent)
  Usage(Usage)
  Completed(CompletedEvent)
  Failed(FailedEvent)
}
