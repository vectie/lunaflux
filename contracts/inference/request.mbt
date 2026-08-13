///|
/// Whether the caller requests token events or one buffered completion.
pub(all) enum StreamPreference {
  Streaming
  Buffered
} derive(Debug, Eq)

///|
/// Cache lookup/insertion authority. Cache sharing remains explicitly scoped.
pub(all) enum CachePermission {
  Disabled
  ReadOnly
  ReadWrite
} derive(Debug, Eq)

///|
/// Scoped cache policy carried by one request.
pub struct CachePolicy {
  scope : CacheScope
  permission : CachePermission
}

///|
pub fn CachePolicy::new(
  scope : CacheScope,
  permission : CachePermission,
) -> CachePolicy {
  { scope, permission }
}

///|
pub fn CachePolicy::scope(self : CachePolicy) -> CacheScope {
  self.scope
}

///|
pub fn CachePolicy::permission(self : CachePolicy) -> CachePermission {
  self.permission
}

///|
/// Relative monotonic request budget, never a wall-clock timestamp.
pub struct DeadlineBudget {
  milliseconds : UInt64
} derive(Debug, Eq)

///|
pub fn DeadlineBudget::from_milliseconds(
  milliseconds : UInt64,
  limits : InferenceLimits,
) -> DeadlineBudget raise InferenceContractError {
  if milliseconds == 0UL || milliseconds > limits.max_deadline_millis() {
    raise invalid(DeadlineBudget, OutOfRange, 0)
  }
  { milliseconds, }
}

///|
pub fn DeadlineBudget::milliseconds(self : DeadlineBudget) -> UInt64 {
  self.milliseconds
}

///|
/// Immutable validated stop conditions. It deliberately has no `Debug` because
/// stop strings are request payload.
pub struct StopConditions {
  token_ids : ReadOnlyArray[Int]
  strings : ReadOnlyArray[String]
}

///|
pub fn StopConditions::new(
  token_ids : ArrayView[Int],
  strings : ArrayView[Bytes],
  limits : InferenceLimits,
) -> StopConditions raise InferenceContractError {
  if token_ids.length() > limits.max_stop_token_ids() {
    raise invalid(StopToken, TooMany, limits.max_stop_token_ids())
  }
  if strings.length() > limits.max_stop_strings() {
    raise invalid(StopString, TooMany, limits.max_stop_strings())
  }
  for index, token_id in token_ids {
    if token_id < 0 || token_id > limits.max_token_id() {
      raise invalid(StopToken, OutOfRange, index)
    }
    for prior in 0..<index {
      if token_ids[prior] == token_id {
        raise invalid(StopToken, Duplicate, index)
      }
    }
  }
  let owned_strings : Array[String] = []
  for index, bytes in strings {
    if bytes.length() == 0 {
      raise invalid(StopString, Empty, index)
    }
    if bytes.length() > limits.max_stop_string_bytes() {
      raise invalid(StopString, TooLong, index)
    }
    let value = @utf8.decode(bytes) catch {
      _ => raise invalid(StopString, InvalidFormat, index)
    }
    for prior in 0..<owned_strings.length() {
      if owned_strings[prior] == value {
        raise invalid(StopString, Duplicate, index)
      }
    }
    owned_strings.push(value)
  }
  {
    token_ids: ReadOnlyArray::from_array(token_ids),
    strings: ReadOnlyArray::from_array(owned_strings),
  }
}

///|
pub fn StopConditions::token_ids(self : StopConditions) -> ReadOnlyArray[Int] {
  self.token_ids
}

///|
/// Returns payload-sensitive stop text for matching only. Callers must not
/// include the returned strings in diagnostics, logs, or metric labels.
pub fn StopConditions::strings(self : StopConditions) -> ReadOnlyArray[String] {
  self.strings
}

///|
fn StopConditions::validate(
  self : StopConditions,
  limits : InferenceLimits,
) -> Unit raise InferenceContractError {
  if self.token_ids.length() > limits.max_stop_token_ids() {
    raise invalid(StopToken, TooMany, limits.max_stop_token_ids())
  }
  if self.strings.length() > limits.max_stop_strings() {
    raise invalid(StopString, TooMany, limits.max_stop_strings())
  }
  for index, token_id in self.token_ids {
    if token_id < 0 || token_id > limits.max_token_id() {
      raise invalid(StopToken, OutOfRange, index)
    }
  }
  for index, value in self.strings {
    if @utf8.encode(value).length() > limits.max_stop_string_bytes() {
      raise invalid(StopString, TooLong, index)
    }
  }
}

///|
/// Canonical, transport-independent native generation request.
///
/// The type deliberately has no `Debug`; access to payload-bearing fields is
/// explicit through typed accessors.
pub struct GenerateRequest {
  protocol_version : ProtocolVersion
  request_id : RequestId
  model_identity : @spec.ModelIdentity
  input : Input
  max_new_tokens : Int
  context_ceiling : Int
  sampling : SamplingParameters
  sampling_seed : SamplingSeed
  stops : StopConditions
  stream_preference : StreamPreference
  deadline : DeadlineBudget
  cache : CachePolicy
  trace : TraceCorrelation?
}

///|
pub fn GenerateRequest::new(
  protocol_version~ : ProtocolVersion,
  request_id~ : RequestId,
  model_identity~ : @spec.ModelIdentity,
  input~ : Input,
  max_new_tokens~ : Int,
  context_ceiling~ : Int,
  sampling~ : SamplingParameters,
  sampling_seed~ : SamplingSeed,
  stops~ : StopConditions,
  stream_preference~ : StreamPreference,
  deadline~ : DeadlineBudget,
  cache~ : CachePolicy,
  trace~ : TraceCorrelation?,
  limits~ : InferenceLimits,
) -> GenerateRequest raise InferenceContractError {
  if protocol_version != ProtocolVersion::v1() {
    raise invalid(ProtocolVersion, Unsupported, 0)
  }
  if max_new_tokens <= 0 || max_new_tokens > limits.max_new_tokens() {
    raise invalid(MaximumNewTokens, OutOfRange, 0)
  }
  if context_ceiling <= 0 || context_ceiling > limits.max_context_tokens() {
    raise invalid(ContextCeiling, OutOfRange, 0)
  }
  if max_new_tokens > context_ceiling {
    raise invalid(ContextCeiling, Inconsistent, 0)
  }
  input.validate(limits)
  match input {
    TokenIds(tokens) if tokens.length() > context_ceiling - max_new_tokens =>
      raise invalid(ContextCeiling, Inconsistent, 0)
    _ => ()
  }
  sampling.validate(limits)
  stops.validate(limits)
  if deadline.milliseconds() > limits.max_deadline_millis() {
    raise invalid(DeadlineBudget, OutOfRange, 0)
  }
  if cache.scope().as_string().length() > limits.max_cache_scope_bytes() {
    raise invalid(CacheScope, TooLong, 0)
  }
  match trace {
    Some(value) if value.as_string().length() > limits.max_trace_bytes() =>
      raise invalid(TraceCorrelation, TooLong, 0)
    _ => ()
  }
  {
    protocol_version,
    request_id,
    model_identity,
    input,
    max_new_tokens,
    context_ceiling,
    sampling,
    sampling_seed,
    stops,
    stream_preference,
    deadline,
    cache,
    trace,
  }
}

///|
pub fn GenerateRequest::protocol_version(
  self : GenerateRequest,
) -> ProtocolVersion {
  self.protocol_version
}

///|
pub fn GenerateRequest::request_id(self : GenerateRequest) -> RequestId {
  self.request_id
}

///|
pub fn GenerateRequest::model_identity(
  self : GenerateRequest,
) -> @spec.ModelIdentity {
  self.model_identity
}

///|
pub fn GenerateRequest::input(self : GenerateRequest) -> Input {
  self.input
}

///|
pub fn GenerateRequest::max_new_tokens(self : GenerateRequest) -> Int {
  self.max_new_tokens
}

///|
pub fn GenerateRequest::context_ceiling(self : GenerateRequest) -> Int {
  self.context_ceiling
}

///|
pub fn GenerateRequest::sampling(self : GenerateRequest) -> SamplingParameters {
  self.sampling
}

///|
pub fn GenerateRequest::sampling_seed(self : GenerateRequest) -> SamplingSeed {
  self.sampling_seed
}

///|
pub fn GenerateRequest::stops(self : GenerateRequest) -> StopConditions {
  self.stops
}

///|
pub fn GenerateRequest::stream_preference(
  self : GenerateRequest,
) -> StreamPreference {
  self.stream_preference
}

///|
pub fn GenerateRequest::deadline(self : GenerateRequest) -> DeadlineBudget {
  self.deadline
}

///|
pub fn GenerateRequest::cache(self : GenerateRequest) -> CachePolicy {
  self.cache
}

///|
pub fn GenerateRequest::trace(self : GenerateRequest) -> TraceCorrelation? {
  self.trace
}
