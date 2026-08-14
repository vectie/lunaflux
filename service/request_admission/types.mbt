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
/// One tokenized scheduler request paired with its exact incremental decoded
/// output owner. The tokenizer and payload-bearing state remain private and
/// the type deliberately has no `Debug`.
pub struct AdmittedRequest {
  priv scheduler_request : @core.TokenizedRequest
  priv output : @incremental_output.IncrementalOutput
  priv tokenizer_digest : @tokenizer.TokenizerDigest
  priv input_tokens : Int
}

///|
pub fn AdmittedRequest::scheduler_request(
  self : AdmittedRequest,
) -> @core.TokenizedRequest {
  self.scheduler_request
}

///|
pub fn AdmittedRequest::tokenizer_digest(
  self : AdmittedRequest,
) -> @tokenizer.TokenizerDigest {
  self.tokenizer_digest
}

///|
pub fn AdmittedRequest::input_token_count(self : AdmittedRequest) -> Int {
  self.input_tokens
}

///|
pub fn AdmittedRequest::is_stop_token(
  self : AdmittedRequest,
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
pub fn AdmittedRequest::push_token_into(
  self : AdmittedRequest,
  token_id : Int,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> @incremental_output.IncrementalOutputStep raise RequestAdmissionError {
  if self.is_stop_token(token_id) {
    raise Invalid(OutputContract)
  }
  self.output.push_token_into(token_id, destination, destination_offset~) catch {
    _ => raise Invalid(OutputContract)
  }
}

///|
pub fn AdmittedRequest::finish_into(
  self : AdmittedRequest,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Int raise RequestAdmissionError {
  self.output.finish_into(destination, destination_offset~) catch {
    _ => raise Invalid(OutputContract)
  }
}

///|
pub fn AdmittedRequest::push_token_into_status(
  self : AdmittedRequest,
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
pub fn AdmittedRequest::finish_into_status(
  self : AdmittedRequest,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> @incremental_output.IncrementalOutputStatus {
  self.output.finish_into_status(destination, destination_offset~)
}

///|
pub fn AdmittedRequest::output_stopped(self : AdmittedRequest) -> Bool {
  self.output.is_stopped()
}

///|
pub fn AdmittedRequest::output_finished(self : AdmittedRequest) -> Bool {
  self.output.is_finished()
}
