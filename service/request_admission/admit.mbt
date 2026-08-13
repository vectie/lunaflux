///|
priv struct PreparedInput {
  tokens : @inference.TokenBuffer
  deadline : @inference.AdmissionDeadline
  effective : @inference.EffectiveLimits
}

///|
fn prepare_input(
  receipt : RequestReceipt,
  request : @inference.GenerateRequest,
  tokenizer : @tokenizer.TokenizerSpec,
  expected_model : @spec.ModelIdentity,
  inference : @inference.InferenceLimits,
) -> PreparedInput raise RequestAdmissionError {
  if request.model_identity() != expected_model {
    raise Invalid(ModelIdentity)
  }
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
  let tokens = match request.input() {
    TokenIds(value) =>
      @inference.TokenBuffer::new(value.token_ids(), inference) catch {
        _ => raise Invalid(Input)
      }
    Text(value) => {
      let encoded = tokenizer.encode_text(
        value.text(),
        special_policy=Reject,
        output_limit=input_capacity,
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
  if tokens.length() > input_capacity {
    raise Invalid(Input)
  }
  { tokens, deadline, effective }
}

///|
fn finish_admission(
  request : @inference.GenerateRequest,
  tokenizer : @tokenizer.TokenizerSpec,
  inference : @inference.InferenceLimits,
  prepared : PreparedInput,
  observed_at_millis : UInt64,
  receipt : RequestReceipt,
) -> AdmittedRequest raise RequestAdmissionError {
  if observed_at_millis < receipt.at_millis {
    raise Invalid(ReceiptTime)
  }
  if prepared.deadline.is_expired(observed_at_millis) {
    raise Invalid(Deadline)
  }
  let no_stop_strings : Array[Bytes] = []
  let scheduler_stops = @inference.StopConditions::new(
    request.stops().token_ids(),
    no_stop_strings,
    inference,
  ) catch {
    _ => raise Invalid(SchedulerContract)
  }
  let output = @incremental_output.IncrementalOutput::new(
    tokenizer,
    request.stops(),
    inference,
    special_policy=Skip,
  ) catch {
    _ => raise Invalid(OutputContract)
  }
  let scheduler_request = @core.TokenizedRequest::new(
    request_id=request.request_id(),
    model_identity=request.model_identity(),
    input_tokens=prepared.tokens,
    sampling=request.sampling(),
    sampling_seed=request.sampling_seed(),
    stops=scheduler_stops,
    cache=request.cache(),
    stream_preference=request.stream_preference(),
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
/// Tokenizes and admits one request against the exact selected model and
/// tokenizer. The clock is sampled after tokenization, so parsing and
/// tokenization consume the budget captured by `RequestReceipt`.
///
/// This synchronous admission function belongs on a bounded tokenizer worker,
/// not an async network reactor.
pub fn admit(
  receipt : RequestReceipt,
  request : @inference.GenerateRequest,
  tokenizer : @tokenizer.TokenizerSpec,
  expected_model : @spec.ModelIdentity,
  inference : @inference.InferenceLimits,
  clock : @monotonic_clock.MonotonicClock,
) -> AdmittedRequest raise RequestAdmissionError {
  let prepared = prepare_input(
    receipt, request, tokenizer, expected_model, inference,
  )
  let observed_at_millis = clock.now_millis() catch {
    _ => raise ClockUnavailable
  }
  finish_admission(
    request, tokenizer, inference, prepared, observed_at_millis, receipt,
  )
}
