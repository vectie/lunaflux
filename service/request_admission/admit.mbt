///|
priv struct PreparedRequest {
  request : @inference.GenerateRequest
  deadline : @inference.AdmissionDeadline
  effective : @inference.EffectiveLimits
  input_capacity : Int
}

///|
priv struct PreparedInput {
  request : @inference.GenerateRequest
  tokens : @inference.TokenBuffer
  deadline : @inference.AdmissionDeadline
  effective : @inference.EffectiveLimits
}

///|
fn receive_with_clock(
  buffer : @framed_wire.RequestFrameBuffer,
  source : FixedArray[Byte],
  length : Int,
  read_clock : () -> UInt64 raise RequestAdmissionError,
) -> ReceivedRequest raise RequestAdmissionError {
  let receipt : RequestReceipt = { at_millis: read_clock() }
  let frame = buffer.load(source, length) catch {
    _ => raise Invalid(FrameContract)
  }
  { receipt, frame, consumed: false }
}

///|
/// Captures the monotonic receipt before parsing the supplied request frame.
/// The returned owner keeps that ordering evidence inseparable from the
/// validated request and can be admitted only once.
pub fn receive(
  buffer : @framed_wire.RequestFrameBuffer,
  source : FixedArray[Byte],
  length : Int,
  clock : @monotonic_clock.MonotonicClock,
) -> ReceivedRequest raise RequestAdmissionError {
  receive_with_clock(buffer, source, length, () => {
    clock.now_millis() catch {
      _ => raise ClockUnavailable
    }
  })
}

///|
fn readmit_request(
  request : @inference.GenerateRequest,
  inference : @inference.InferenceLimits,
) -> @inference.GenerateRequest raise RequestAdmissionError {
  @inference.GenerateRequest::new(
    protocol_version=request.protocol_version(),
    request_id=request.request_id(),
    model_identity=request.model_identity(),
    input=request.input(),
    max_new_tokens=request.max_new_tokens(),
    context_ceiling=request.context_ceiling(),
    sampling=request.sampling(),
    sampling_seed=request.sampling_seed(),
    stops=request.stops(),
    stream_preference=request.stream_preference(),
    deadline=request.deadline(),
    cache=request.cache(),
    trace=request.trace(),
    limits=inference,
  ) catch {
    _ => raise Invalid(RequestContract)
  }
}

///|
fn preflight_request(
  receipt : RequestReceipt,
  request : @inference.GenerateRequest,
  tokenizer : @tokenizer.TokenizerSpec,
  expected_tokenizer : @tokenizer.TokenizerDigest,
  expected_model : @spec.ModelIdentity,
  inference : @inference.InferenceLimits,
) -> PreparedRequest raise RequestAdmissionError {
  if request.model_identity() != expected_model {
    raise Invalid(ModelIdentity)
  }
  if tokenizer.digest() != expected_tokenizer {
    raise Invalid(TokenizerIdentity)
  }
  let request = readmit_request(request, inference)
  let deadline = @inference.AdmissionDeadline::from_receipt(
    receipt.at_millis,
    request.deadline(),
  ) catch {
    _ => raise Invalid(Deadline)
  }
  let effective = @inference.EffectiveLimits::new(
    max_new_tokens=request.max_new_tokens(),
    context_ceiling=request.context_ceiling(),
    inference,
  ) catch {
    _ => raise Invalid(Input)
  }
  let input_capacity = effective.context_ceiling() - effective.max_new_tokens()
  if input_capacity <= 0 {
    raise Invalid(Input)
  }
  { request, deadline, effective, input_capacity }
}

///|
fn tokenize_input(
  prepared : PreparedRequest,
  tokenizer : @tokenizer.TokenizerSpec,
  inference : @inference.InferenceLimits,
) -> PreparedInput raise RequestAdmissionError {
  let tokens = match prepared.request.input() {
    TokenIds(value) =>
      @inference.TokenBuffer::new(value.token_ids(), inference) catch {
        _ => raise Invalid(Input)
      }
    Text(value) => {
      let encoded = tokenizer.encode_text(
        value.text(),
        special_policy=Reject,
        output_limit=prepared.input_capacity,
        truncation=RejectOverflow,
      ) catch {
        _ => raise Invalid(Tokenization)
      }
      if encoded.truncated {
        raise Invalid(Tokenization)
      }
      @inference.TokenBuffer::new(encoded.token_ids, inference) catch {
        _ => raise Invalid(Input)
      }
    }
  }
  if tokens.length() > prepared.input_capacity {
    raise Invalid(Input)
  }
  {
    request: prepared.request,
    tokens,
    deadline: prepared.deadline,
    effective: prepared.effective,
  }
}

///|
fn validate_observed_time(
  receipt : RequestReceipt,
  deadline : @inference.AdmissionDeadline,
  observed_at_millis : UInt64,
) -> Unit raise RequestAdmissionError {
  if observed_at_millis < receipt.at_millis {
    raise Invalid(ReceiptTime)
  }
  if deadline.is_expired(observed_at_millis) {
    raise Invalid(Deadline)
  }
}

///|
fn finish_admission(
  tokenizer : @tokenizer.TokenizerSpec,
  inference : @inference.InferenceLimits,
  prepared : PreparedInput,
  observed_at_millis : UInt64,
  receipt : RequestReceipt,
) -> AdmittedRequest raise RequestAdmissionError {
  validate_observed_time(receipt, prepared.deadline, observed_at_millis)
  let no_stop_strings : Array[Bytes] = []
  let scheduler_stops = @inference.StopConditions::new(
    prepared.request.stops().token_ids(),
    no_stop_strings,
    inference,
  ) catch {
    _ => raise Invalid(SchedulerContract)
  }
  let output = @incremental_output.IncrementalOutput::new(
    tokenizer,
    prepared.request.stops(),
    inference,
    special_policy=Skip,
  ) catch {
    _ => raise Invalid(OutputContract)
  }
  let scheduler_request = @core.TokenizedRequest::new(
    request_id=prepared.request.request_id(),
    model_identity=prepared.request.model_identity(),
    input_tokens=prepared.tokens,
    sampling=prepared.request.sampling(),
    sampling_seed=prepared.request.sampling_seed(),
    stops=scheduler_stops,
    cache=prepared.request.cache(),
    stream_preference=prepared.request.stream_preference(),
    effective_limits=prepared.effective,
    admission_deadline=prepared.deadline,
  ) catch {
    _ => raise Invalid(SchedulerContract)
  }
  {
    scheduler_request,
    output,
    tokenizer_digest: tokenizer.digest(),
    input_tokens: prepared.tokens.length(),
  }
}

///|
fn admit_with_clock(
  receipt : RequestReceipt,
  request : @inference.GenerateRequest,
  tokenizer : @tokenizer.TokenizerSpec,
  expected_tokenizer : @tokenizer.TokenizerDigest,
  expected_model : @spec.ModelIdentity,
  inference : @inference.InferenceLimits,
  read_clock : () -> UInt64 raise RequestAdmissionError,
) -> AdmittedRequest raise RequestAdmissionError {
  let prepared_request = preflight_request(
    receipt, request, tokenizer, expected_tokenizer, expected_model, inference,
  )
  validate_observed_time(receipt, prepared_request.deadline, read_clock())
  let prepared = tokenize_input(prepared_request, tokenizer, inference)
  let observed_at_millis = read_clock()
  finish_admission(tokenizer, inference, prepared, observed_at_millis, receipt)
}

///|
/// Tokenizes and admits one request against the exact selected model and
/// independently expected tokenizer digest. The clock is sampled after cheap
/// canonical preflight and again after tokenization: already-expired work is
/// rejected before BPE, while tokenization still consumes the original budget.
///
/// This synchronous admission function belongs on a bounded tokenizer worker,
/// not an async network reactor.
pub fn admit(
  received : ReceivedRequest,
  tokenizer : @tokenizer.TokenizerSpec,
  expected_tokenizer : @tokenizer.TokenizerDigest,
  expected_model : @spec.ModelIdentity,
  inference : @inference.InferenceLimits,
  clock : @monotonic_clock.MonotonicClock,
) -> AdmittedRequest raise RequestAdmissionError {
  if received.consumed {
    raise Invalid(FrameContract)
  }
  received.consumed = true
  let request = received.frame.request() catch {
    _ => raise Invalid(FrameContract)
  }
  admit_with_clock(
    received.receipt,
    request,
    tokenizer,
    expected_tokenizer,
    expected_model,
    inference,
    () => clock.now_millis() catch { _ => raise ClockUnavailable },
  )
}
