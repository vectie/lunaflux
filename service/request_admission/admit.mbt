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
fn preflight_bound_request(
  deadline : @inference.AdmissionDeadline,
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
  let effective = @inference.EffectiveLimits::new(
    max_new_tokens=request.max_new_tokens(),
    context_ceiling=request.context_ceiling(),
    inference,
  ) catch {
    _ => raise Invalid(Input)
  }
  let input_capacity = Int::min(
    effective.context_ceiling() - effective.max_new_tokens(),
    Int::min(inference.max_input_tokens(), tokenizer.limits().max_output_tokens),
  )
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
    TokenIds(value) => {
      if !value.is_live() {
        raise Invalid(Input)
      }
      value
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
fn finish_preparation(
  tokenizer : @tokenizer.TokenizerSpec,
  inference : @inference.InferenceLimits,
  prepared : PreparedInput,
  observed_at_millis : UInt64,
  receipt : RequestReceipt,
) -> LunaPreparedRequest raise RequestAdmissionError {
  validate_observed_time(receipt, prepared.deadline, observed_at_millis)
  let scheduler_stops = prepared.request.stops().token_only()
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
  let claim : LunaPreparedRequestClaim = {
    scheduler_request: Some(scheduler_request),
    detached_output: Some(output),
    pooled_output: None,
    token_lease: None,
    pool: None,
    lane: -1,
    generation: 0UL,
    tokenizer_digest: tokenizer.digest(),
    inference_limits: inference,
    input_tokens: prepared.tokens.length(),
  }
  { claim: Some(claim) }
}

///|
fn prepare_bound_with_clock(
  receipt : RequestReceipt,
  deadline : @inference.AdmissionDeadline,
  request : @inference.GenerateRequest,
  tokenizer : @tokenizer.TokenizerSpec,
  expected_tokenizer : @tokenizer.TokenizerDigest,
  expected_model : @spec.ModelIdentity,
  inference : @inference.InferenceLimits,
  read_clock : () -> UInt64 raise RequestAdmissionError,
) -> LunaPreparedRequest raise RequestAdmissionError {
  let prepared_request = preflight_bound_request(
    deadline, request, tokenizer, expected_tokenizer, expected_model, inference,
  )
  validate_observed_time(receipt, prepared_request.deadline, read_clock())
  let prepared = tokenize_input(prepared_request, tokenizer, inference)
  let observed_at_millis = read_clock()
  finish_preparation(
    tokenizer, inference, prepared, observed_at_millis, receipt,
  )
}

///|
/// Prepares one request against the exact selected model and independently
/// expected tokenizer digest. The clock is sampled after cheap canonical
/// preflight and again after tokenization: already-expired work is rejected
/// before BPE, while tokenization still consumes the original budget.
///
/// This synchronous admission function belongs on a bounded tokenizer worker,
/// not an async network reactor.
pub fn prepare_luna_request(
  received : ReceivedRequest,
  tokenizer : @tokenizer.TokenizerSpec,
  expected_tokenizer : @tokenizer.TokenizerDigest,
  expected_model : @spec.ModelIdentity,
  inference : @inference.InferenceLimits,
  clock : @monotonic_clock.MonotonicClock,
) -> LunaPreparedRequest raise RequestAdmissionError {
  if received.consumed {
    raise Invalid(FrameContract)
  }
  received.consumed = true
  prepare_bound_with_clock(
    received.receipt,
    received.deadline,
    received.request,
    tokenizer,
    expected_tokenizer,
    expected_model,
    inference,
    () => clock.now_millis() catch { _ => raise ClockUnavailable },
  )
}
