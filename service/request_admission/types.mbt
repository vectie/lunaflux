///|
pub(all) enum RequestAdmissionRule {
  ModelIdentity
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
/// Opaque monotonic timestamp captured at the first trusted frame-receipt
/// boundary. Callers cannot manufacture or inspect its process-local value.
#valtype
pub struct RequestReceipt {
  priv at_millis : UInt64
}

///|
/// Captures request receipt before frame parsing or tokenization begins.
pub fn RequestReceipt::capture(
  clock : @monotonic_clock.MonotonicClock,
) -> RequestReceipt raise RequestAdmissionError {
  let at_millis = clock.now_millis() catch { _ => raise ClockUnavailable }
  { at_millis, }
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
  for candidate in self.scheduler_request.stops().token_ids() {
    if candidate == token_id {
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
pub fn AdmittedRequest::output_stopped(self : AdmittedRequest) -> Bool {
  self.output.is_stopped()
}

///|
pub fn AdmittedRequest::output_finished(self : AdmittedRequest) -> Bool {
  self.output.is_finished()
}
