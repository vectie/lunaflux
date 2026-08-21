///|
/// Tokenized, immutable request accepted by the single scheduler owner.
///
/// This is deliberately not constructible from text and has no `Debug`:
/// tokenization and transport parsing are outer-layer responsibilities, while
/// token IDs remain payload-sensitive. Stop strings and cache semantics never
/// cross the scheduler's token-only projection boundary.
pub struct TokenizedRequest {
  priv request_id : @inference.RequestId
  priv model_identity : @spec.ModelIdentity
  priv input_tokens : @inference.TokenBuffer
  priv sampling : @inference.SamplingParameters
  priv sampling_seed : @inference.SamplingSeed
  priv stop_tokens : @inference.LunaRequestStopTokenView
  priv stream_preference : @inference.StreamPreference
  priv effective_limits : @inference.EffectiveLimits
  priv admission_deadline : @inference.AdmissionDeadline
}

///|
pub fn TokenizedRequest::new(
  request_id~ : @inference.RequestId,
  model_identity~ : @spec.ModelIdentity,
  input_tokens~ : @inference.TokenBuffer,
  sampling~ : @inference.SamplingParameters,
  sampling_seed~ : @inference.SamplingSeed,
  stop_tokens~ : @inference.LunaRequestStopTokenView,
  stream_preference~ : @inference.StreamPreference,
  effective_limits~ : @inference.EffectiveLimits,
  admission_deadline~ : @inference.AdmissionDeadline,
) -> TokenizedRequest raise SchedulerError {
  if !input_tokens.is_live() {
    raise invalid(Request, Stale)
  }
  if !stop_tokens.is_live() {
    raise invalid(Request, Stale)
  }
  if input_tokens.length() >
    effective_limits.context_ceiling() - effective_limits.max_new_tokens() {
    raise invalid(Context, OutOfRange)
  }
  {
    request_id,
    model_identity,
    input_tokens,
    sampling,
    sampling_seed,
    stop_tokens,
    stream_preference,
    effective_limits,
    admission_deadline,
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
fn TokenizedRequest::input_tokens(
  self : TokenizedRequest,
) -> @inference.TokenBuffer {
  self.input_tokens
}

///|
fn TokenizedRequest::input_maximum_token_status(
  self : TokenizedRequest,
  maximum_token_id : Int,
) -> @inference.LunaTokenBufferBoundStatus {
  self.input_tokens.maximum_token_status(maximum_token_id)
}

///|
fn TokenizedRequest::input_token_at(
  self : TokenizedRequest,
  index : Int,
) -> Int raise SchedulerError {
  let status = self.input_tokens.token_status(index)
  if !status.is_ready() {
    raise invalid(Request, Stale)
  }
  status.token_id()
}

///|
pub fn TokenizedRequest::sampling(
  self : TokenizedRequest,
) -> @inference.SamplingParameters {
  self.sampling
}

///|
pub fn TokenizedRequest::sampling_seed(
  self : TokenizedRequest,
) -> @inference.SamplingSeed {
  self.sampling_seed
}

///|
fn TokenizedRequest::semantic_stop_tokens_live(self : TokenizedRequest) -> Bool {
  self.stop_tokens.is_live()
}

///|
fn TokenizedRequest::retain_stop_tokens_into(
  self : TokenizedRequest,
  slot : @inference.LunaRequestStopTokenRetentionSlot,
) -> Unit raise SchedulerError {
  if !self.stop_tokens.is_live() {
    raise invalid(Request, Stale)
  }
  try self.stop_tokens.retain_into(slot) catch {
    _ => raise invalid(Request, Backpressured)
  } noraise {
    _ => ()
  }
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
pub fn TokenizedRequest::admission_deadline(
  self : TokenizedRequest,
) -> @inference.AdmissionDeadline {
  self.admission_deadline
}
