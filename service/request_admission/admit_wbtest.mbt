///|
fn fixture_inference() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=64,
    max_new_tokens=16,
    max_context_tokens=64,
    max_token_id=255,
    max_stop_token_ids=4,
    max_stop_strings=4,
    max_stop_string_bytes=16,
    max_trace_bytes=16,
    max_cache_scope_bytes=16,
    max_decoded_delta_bytes=32,
    max_deadline_millis=1000UL,
    max_top_k=256,
    max_temperature=4.0,
  )
}

///|
fn fixture_identity(seed : String) -> @spec.ModelIdentity raise {
  @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(seed.repeat(64)),
    @spec.PlanDigest::from_sha256(
      if seed == "0" {
        "1".repeat(64)
      } else {
        "2".repeat(64)
      },
    ),
  )
}

///|
fn fixture_tokenizer() -> @tokenizer.TokenizerSpec raise {
  let vocabulary : Array[@tokenizer.VocabularyEntry] = []
  for value in 0..<256 {
    vocabulary.push({ token_id: value, piece: [value.to_byte()], kind: Normal })
  }
  @tokenizer.TokenizerSpec::new(
    vocabulary,
    [],
    @tokenizer.TokenizerDigest::parse("0123456789abcdef".repeat(4)),
    {
      max_input_bytes: 64,
      max_output_tokens: 64,
      max_decoded_bytes: 256,
      max_vocab_entries: 256,
      max_merge_rules: 4,
      max_token_bytes: 1,
      max_special_tokens: 1,
    },
  )
}

///|
fn fixture_request(
  identity : @spec.ModelIdentity,
  deadline_millis? : UInt64 = 100UL,
) -> @inference.GenerateRequest raise {
  let limits = fixture_inference()
  @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(7UL),
    model_identity=identity,
    input=@inference.Input::from_utf8(b"abc", limits),
    max_new_tokens=4,
    context_ceiling=16,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(9UL),
    stops=@inference.StopConditions::new([9], [b"END"], limits),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(
      deadline_millis, limits,
    ),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"scope", limits),
      Disabled,
    ),
    trace=None,
    limits~,
  )
}

///|
test "public receipt capture and admission use the monotonic clock" {
  let clock = @monotonic_clock.MonotonicClock::system()
  let receipt = RequestReceipt::capture(clock)
  let identity = fixture_identity("0")
  let admitted = admit(
    receipt,
    fixture_request(identity, deadline_millis=1000UL),
    fixture_tokenizer(),
    identity,
    fixture_inference(),
    clock,
  )
  assert_eq(admitted.scheduler_request().request_id().value(), 7UL)
}

///|
fn admit_at(
  receipt : RequestReceipt,
  request : @inference.GenerateRequest,
  tokenizer : @tokenizer.TokenizerSpec,
  expected_model : @spec.ModelIdentity,
  inference : @inference.InferenceLimits,
  observed_at_millis : UInt64,
) -> AdmittedRequest raise RequestAdmissionError {
  let prepared = prepare_input(
    receipt, request, tokenizer, expected_model, inference,
  )
  finish_admission(
    request, tokenizer, inference, prepared, observed_at_millis, receipt,
  )
}

///|
test "text admission preserves receipt deadline and separates string stops" {
  let identity = fixture_identity("0")
  let tokenizer = fixture_tokenizer()
  let admitted = admit_at(
    { at_millis: 100UL },
    fixture_request(identity),
    tokenizer,
    identity,
    fixture_inference(),
    150UL,
  )
  let scheduler_request = admitted.scheduler_request()
  assert_eq(scheduler_request.input_tokens().length(), 3)
  assert_eq(scheduler_request.admission_deadline().at_millis(), 200UL)
  assert_eq(scheduler_request.stops().token_ids().length(), 1)
  assert_eq(scheduler_request.stops().strings().length(), 0)
  assert_eq(admitted.input_token_count(), 3)
  assert_eq(admitted.tokenizer_digest(), tokenizer.digest())
  let destination = FixedArray::make(8, b'!')
  assert_true(admitted.is_stop_token(9))
  try admitted.push_token_into(9, destination, destination_offset=1) catch {
    Invalid(OutputContract) => ()
    _ => fail("unexpected stop-token output error")
  } noraise {
    _ => fail("stop token was decoded as ordinary output")
  }
  let step = admitted.push_token_into(65, destination, destination_offset=1)
  assert_eq(step.copied_bytes(), 1)
  assert_eq(destination[1], b'A')
  assert_eq(admitted.finish_into(destination, destination_offset=2), 0)
  assert_true(admitted.output_finished())
}

///|
test "identity receipt and expired budget reject before publication" {
  let identity = fixture_identity("0")
  let request = fixture_request(identity)
  try
    admit_at(
      { at_millis: 100UL },
      request,
      fixture_tokenizer(),
      fixture_identity("f"),
      fixture_inference(),
      101UL,
    )
  catch {
    Invalid(ModelIdentity) => ()
    _ => fail("unexpected model binding error")
  } noraise {
    _ => fail("foreign model was admitted")
  }
  for case in [(99UL, ReceiptTime), (200UL, Deadline)] {
    let (observed, expected) = case
    try
      admit_at(
        { at_millis: 100UL },
        request,
        fixture_tokenizer(),
        identity,
        fixture_inference(),
        observed,
      )
    catch {
      Invalid(rule) => assert_eq(rule, expected)
      _ => fail("unexpected time admission error")
    } noraise {
      _ => fail("invalid admission time was accepted")
    }
  }
}

///|
test "string stops remain in incremental output instead of scheduler state" {
  let identity = fixture_identity("0")
  let admitted = admit_at(
    { at_millis: 10UL },
    fixture_request(identity),
    fixture_tokenizer(),
    identity,
    fixture_inference(),
    11UL,
  )
  let destination = FixedArray::make(8, b'!')
  assert_eq(
    admitted
    .push_token_into(69, destination, destination_offset=0)
    .copied_bytes(),
    0,
  )
  assert_eq(
    admitted
    .push_token_into(78, destination, destination_offset=0)
    .copied_bytes(),
    0,
  )
  let matched = admitted.push_token_into(68, destination, destination_offset=0)
  assert_eq(matched.copied_bytes(), 0)
  assert_true(matched.stop_matched())
  assert_true(admitted.output_stopped())
  try admitted.finish_into(destination, destination_offset=0) catch {
    Invalid(OutputContract) => ()
    _ => fail("unexpected stopped-output terminal error")
  } noraise {
    _ => fail("matched stop suffix was flushed")
  }
}
