///|
/// Tokenized, immutable request accepted by the single scheduler owner.
///
/// This is deliberately not constructible from text and has no `Debug`:
/// tokenization and transport parsing are outer-layer responsibilities, while
/// token IDs and stop strings remain payload-sensitive.
pub struct TokenizedRequest {
  request_id : @inference.RequestId
  model_identity : @spec.ModelIdentity
  input_tokens : @inference.TokenBuffer
  sampling : @inference.SamplingParameters
  stops : @inference.StopConditions
  cache : @inference.CachePolicy
  stream_preference : @inference.StreamPreference
  effective_limits : @inference.EffectiveLimits
  deadline_budget : @inference.DeadlineBudget
}

///|
pub fn TokenizedRequest::new(
  request_id~ : @inference.RequestId,
  model_identity~ : @spec.ModelIdentity,
  input_tokens~ : @inference.TokenBuffer,
  sampling~ : @inference.SamplingParameters,
  stops~ : @inference.StopConditions,
  cache~ : @inference.CachePolicy,
  stream_preference~ : @inference.StreamPreference,
  effective_limits~ : @inference.EffectiveLimits,
  deadline_budget~ : @inference.DeadlineBudget,
) -> TokenizedRequest raise SchedulerError {
  if input_tokens.length() >
    effective_limits.context_ceiling() - effective_limits.max_new_tokens() {
    raise invalid(Context, OutOfRange)
  }
  if stops.strings().length() > 0 {
    raise invalid(Request, Unsupported)
  }
  {
    request_id,
    model_identity,
    input_tokens,
    sampling,
    stops,
    cache,
    stream_preference,
    effective_limits,
    deadline_budget,
  }
}

///|
pub fn TokenizedRequest::request_id(
  self : TokenizedRequest,
) -> @inference.RequestId {
  self.request_id
}

///|
pub fn TokenizedRequest::model_identity(
  self : TokenizedRequest,
) -> @spec.ModelIdentity {
  self.model_identity
}

///|
pub fn TokenizedRequest::input_tokens(
  self : TokenizedRequest,
) -> @inference.TokenBuffer {
  self.input_tokens
}

///|
pub fn TokenizedRequest::sampling(
  self : TokenizedRequest,
) -> @inference.SamplingParameters {
  self.sampling
}

///|
pub fn TokenizedRequest::stops(
  self : TokenizedRequest,
) -> @inference.StopConditions {
  self.stops
}

///|
pub fn TokenizedRequest::cache(
  self : TokenizedRequest,
) -> @inference.CachePolicy {
  self.cache
}

///|
pub fn TokenizedRequest::stream_preference(
  self : TokenizedRequest,
) -> @inference.StreamPreference {
  self.stream_preference
}

///|
pub fn TokenizedRequest::effective_limits(
  self : TokenizedRequest,
) -> @inference.EffectiveLimits {
  self.effective_limits
}

///|
pub fn TokenizedRequest::deadline_budget(
  self : TokenizedRequest,
) -> @inference.DeadlineBudget {
  self.deadline_budget
}
