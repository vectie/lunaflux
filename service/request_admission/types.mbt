///|
pub(all) enum RequestAdmissionRule {
  ModelIdentity
  TokenizerIdentity
  RequestContract
  FrameContract
  ReceiptTime
  Deadline
  Input
  Tokenization
  SchedulerContract
  OutputContract
  LunaPreparedLifecycle
} derive(Debug, Eq)

///|
/// Payload-safe admission failures. Request text, token IDs, stop strings,
/// cache scope, trace data, and tokenizer internals never enter the error.
pub(all) suberror RequestAdmissionError {
  Invalid(RequestAdmissionRule)
  ClockUnavailable
} derive(Debug, Eq)

///|
/// Monotonic timestamp captured inside the trusted frame-receipt operation.
#valtype
priv struct RequestReceipt {
  at_millis : UInt64
}

///|
/// One trusted incremental receiver. The first nonempty append captures its
/// monotonic receipt before the framed reader can copy any byte.
pub struct IncrementalRequestReceiver {
  priv reader : @framed_wire.IncrementalRequestReader
  priv clock : @monotonic_clock.MonotonicClock
  priv mut receipt_at_millis : UInt64
  priv mut has_receipt : Bool
  priv mut terminal : Bool
}

///|
/// One immutable request paired with the monotonic receipt and absolute
/// deadline derived during exact frame binding. The owner is admitted once.
pub struct ReceivedRequest {
  priv receipt : RequestReceipt
  priv request : @inference.GenerateRequest
  priv deadline : @inference.AdmissionDeadline
  priv mut consumed : Bool
}

///|
/// One revocable shell around a preallocated prepared-request claim. Retained
/// aliases observe `None` after the claim is taken, so scheduler and output
/// authority cannot be extracted twice. The type deliberately has no `Debug`.
pub struct LunaPreparedRequest {
  priv mut claim : LunaPreparedRequestClaim?
}

///|
/// Exact scheduler and incremental-output authority transferred from one
/// Luna-prepared shell. Receipt, deadline, tokenizer, and payload-bearing
/// state remain private and the type deliberately has no `Debug`.
pub struct LunaPreparedRequestClaim {
  priv scheduler_request : @core.TokenizedRequest
  priv output : @incremental_output.IncrementalOutput
  priv tokenizer_digest : @tokenizer.TokenizerDigest
  priv inference_limits : @inference.InferenceLimits
  priv input_tokens : Int
}

///|
fn LunaPreparedRequest::require_claim(
  self : LunaPreparedRequest,
) -> LunaPreparedRequestClaim raise RequestAdmissionError {
  match self.claim {
    Some(claim) => claim
    None => raise Invalid(LunaPreparedLifecycle)
  }
}

///|
pub fn LunaPreparedRequest::take_claim(
  self : LunaPreparedRequest,
) -> LunaPreparedRequestClaim raise RequestAdmissionError {
  let claim = self.require_claim()
  self.claim = None
  claim
}

///|
pub fn LunaPreparedRequest::input_token_count(
  self : LunaPreparedRequest,
) -> Int raise RequestAdmissionError {
  self.require_claim().input_tokens
}

///|
pub fn LunaPreparedRequest::tokenizer_digest(
  self : LunaPreparedRequest,
) -> @tokenizer.TokenizerDigest raise RequestAdmissionError {
  self.require_claim().tokenizer_digest
}

///|
pub fn LunaPreparedRequest::inference_limits(
  self : LunaPreparedRequest,
) -> @inference.InferenceLimits raise RequestAdmissionError {
  self.require_claim().inference_limits
}

///|
pub fn LunaPreparedRequest::model_identity(
  self : LunaPreparedRequest,
) -> @spec.ModelIdentity raise RequestAdmissionError {
  self.require_claim().scheduler_request.model_identity()
}

///|
pub fn LunaPreparedRequest::effective_limits(
  self : LunaPreparedRequest,
) -> @inference.EffectiveLimits raise RequestAdmissionError {
  self.require_claim().scheduler_request.effective_limits()
}

///|
pub fn LunaPreparedRequest::stream_preference(
  self : LunaPreparedRequest,
) -> @inference.StreamPreference raise RequestAdmissionError {
  self.require_claim().scheduler_request.stream_preference()
}

///|
pub fn LunaPreparedRequestClaim::scheduler_request(
  self : LunaPreparedRequestClaim,
) -> @core.TokenizedRequest {
  self.scheduler_request
}

///|
pub fn LunaPreparedRequestClaim::is_stop_token(
  self : LunaPreparedRequestClaim,
  token_id : Int,
) -> Bool {
  let candidates = self.scheduler_request.stops().token_ids()
  for index in 0..<candidates.length() {
    if candidates[index] == token_id {
      return true
    }
  }
  false
}

///|
pub fn LunaPreparedRequestClaim::push_token_into_status(
  self : LunaPreparedRequestClaim,
  token_id : Int,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> @incremental_output.IncrementalOutputStatus {
  if self.is_stop_token(token_id) {
    return @incremental_output.IncrementalOutputStatus::rejected()
  }
  self.output.push_token_status(token_id, destination, destination_offset~)
}

///|
pub fn LunaPreparedRequestClaim::finish_into_status(
  self : LunaPreparedRequestClaim,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> @incremental_output.IncrementalOutputStatus {
  self.output.finish_into_status(destination, destination_offset~)
}
