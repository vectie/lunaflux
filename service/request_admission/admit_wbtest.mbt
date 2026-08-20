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
fn fixture_tokenizer_with_digest(
  digest : String,
) -> @tokenizer.TokenizerSpec raise {
  let vocabulary : Array[@tokenizer.VocabularyEntry] = []
  for value in 0..<256 {
    vocabulary.push({ token_id: value, piece: [value.to_byte()], kind: Normal })
  }
  @tokenizer.TokenizerSpec::new(
    vocabulary,
    [],
    @tokenizer.TokenizerDigest::parse(digest),
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
fn fixture_received_source(
  request : @inference.GenerateRequest,
) -> (FixedArray[Byte], Int, @framed_wire.RequestFrameBuffer) raise {
  let limits = @framed_wire.FramedWireLimits::new(
    fixture_inference(),
    max_frame_bytes=4096,
  )
  let frame = @framed_wire.RequestFrameBuffer::new(limits).encode(request)
  let source = FixedArray::make(frame.length(), b'\x00')
  frame.copy_to(source, destination_offset=0)
  (source, frame.length(), @framed_wire.RequestFrameBuffer::new(limits))
}

///|
test "public framed receipt and Luna preparation use the monotonic clock" {
  let clock = @monotonic_clock.MonotonicClock::system()
  let identity = fixture_identity("0")
  let (source, length, buffer) = fixture_received_source(
    fixture_request(identity, deadline_millis=1000UL),
  )
  let received = receive(buffer, source, length, clock)
  let tokenizer = fixture_tokenizer()
  let prepared = prepare_luna_request(
    received,
    tokenizer,
    tokenizer.digest(),
    identity,
    fixture_inference(),
    clock,
  )
  assert_eq(prepared.input_token_count(), 3)
  assert_eq(prepared.tokenizer_digest(), tokenizer.digest())
  assert_eq(prepared.inference_limits(), fixture_inference())
  let retained = prepared
  let claim = prepared.take_claim()
  assert_eq(claim.scheduler_request().request_id().value(), 7UL)
  try retained.take_claim() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("unexpected prepared-request replay error")
  } noraise {
    _ => fail("prepared request was claimed twice")
  }
  try retained.input_token_count() catch {
    Invalid(LunaPreparedLifecycle) => ()
    _ => fail("unexpected retained-shell getter error")
  } noraise {
    _ => fail("retained prepared shell kept read authority")
  }
  try
    prepare_luna_request(
      received,
      tokenizer,
      tokenizer.digest(),
      identity,
      fixture_inference(),
      clock,
    )
  catch {
    Invalid(FrameContract) => ()
    _ => fail("unexpected received-request replay error")
  } noraise {
    _ => fail("received request was prepared twice")
  }
}

///|
fn fixture_deadline(
  receipt : RequestReceipt,
  request : @inference.GenerateRequest,
) -> @inference.AdmissionDeadline raise RequestAdmissionError {
  @inference.AdmissionDeadline::from_receipt(
    receipt.at_millis,
    request.deadline(),
  ) catch {
    _ => raise Invalid(Deadline)
  }
}

///|
test "tokenizer digest substitution rejects before clock or tokenization" {
  let identity = fixture_identity("0")
  let tokenizer = fixture_tokenizer()
  let foreign = fixture_tokenizer_with_digest("fedcba9876543210".repeat(4))
  let clock_reads = Ref(0)
  let receipt : RequestReceipt = { at_millis: 10UL }
  let request = fixture_request(identity)
  try
    prepare_bound_with_clock(
      receipt,
      fixture_deadline(receipt, request),
      request,
      tokenizer,
      foreign.digest(),
      identity,
      fixture_inference(),
      () => {
        clock_reads.val += 1
        11UL
      },
    )
  catch {
    Invalid(TokenizerIdentity) => ()
    _ => fail("unexpected tokenizer binding error")
  } noraise {
    _ => fail("foreign tokenizer digest was accepted")
  }
  assert_eq(clock_reads.val, 0)
}

///|
fn prepare_at(
  receipt : RequestReceipt,
  request : @inference.GenerateRequest,
  tokenizer : @tokenizer.TokenizerSpec,
  expected_tokenizer : @tokenizer.TokenizerDigest,
  expected_model : @spec.ModelIdentity,
  inference : @inference.InferenceLimits,
  observed_at_millis : UInt64,
) -> LunaPreparedRequest raise RequestAdmissionError {
  let prepared_request = preflight_bound_request(
    fixture_deadline(receipt, request),
    request,
    tokenizer,
    expected_tokenizer,
    expected_model,
    inference,
  )
  validate_observed_time(receipt, prepared_request.deadline, observed_at_millis)
  let prepared = tokenize_input(prepared_request, tokenizer, inference)
  finish_preparation(
    tokenizer, inference, prepared, observed_at_millis, receipt,
  )
}

///|
test "text admission preserves receipt deadline and separates string stops" {
  let identity = fixture_identity("0")
  let tokenizer = fixture_tokenizer()
  let prepared = prepare_at(
    { at_millis: 100UL },
    fixture_request(identity),
    tokenizer,
    tokenizer.digest(),
    identity,
    fixture_inference(),
    150UL,
  )
  assert_eq(prepared.input_token_count(), 3)
  let claim = prepared.take_claim()
  let scheduler_request = claim.scheduler_request()
  assert_eq(scheduler_request.input_tokens().length(), 3)
  assert_eq(scheduler_request.admission_deadline().at_millis(), 200UL)
  assert_eq(scheduler_request.stops().token_ids().length(), 1)
  assert_eq(scheduler_request.stops().strings().length(), 0)
  let destination = FixedArray::make(8, b'!')
  assert_true(claim.is_stop_token(9))
  assert_false(
    claim
    .push_token_into_status(9, destination, destination_offset=1)
    .is_ready(),
  )
  let step = claim.push_token_into_status(65, destination, destination_offset=1)
  assert_true(step.is_ready())
  assert_eq(step.copied_bytes(), 1)
  assert_eq(destination[1], b'A')
  let finished = claim.finish_into_status(destination, destination_offset=2)
  assert_true(finished.is_ready())
  assert_eq(finished.copied_bytes(), 0)
}

///|
test "identity receipt and expired budget reject before publication" {
  let identity = fixture_identity("0")
  let request = fixture_request(identity)
  try
    prepare_at(
      { at_millis: 100UL },
      request,
      fixture_tokenizer(),
      fixture_tokenizer().digest(),
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
      prepare_at(
        { at_millis: 100UL },
        request,
        fixture_tokenizer(),
        fixture_tokenizer().digest(),
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
test "request evidence is canonically readmitted under instance limits" {
  let loose = @inference.InferenceLimits::new(
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
    max_deadline_millis=2000UL,
    max_top_k=256,
    max_temperature=4.0,
  )
  let identity = fixture_identity("0")
  let request = @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(8UL),
    model_identity=identity,
    input=@inference.Input::from_token_ids([1, 2, 3], loose),
    max_new_tokens=4,
    context_ceiling=16,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(8UL),
    stops=@inference.StopConditions::new([], [], loose),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(1500UL, loose),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"scope", loose),
      Disabled,
    ),
    trace=None,
    limits=loose,
  )
  try
    prepare_at(
      { at_millis: 0UL },
      request,
      fixture_tokenizer(),
      fixture_tokenizer().digest(),
      identity,
      fixture_inference(),
      1UL,
    )
  catch {
    Invalid(RequestContract) => ()
    _ => fail("unexpected foreign-limit error")
  } noraise {
    _ => fail("request admitted with a foreign deadline ceiling")
  }
}

///|
test "string stops remain in incremental output instead of scheduler state" {
  let identity = fixture_identity("0")
  let prepared = prepare_at(
    { at_millis: 10UL },
    fixture_request(identity),
    fixture_tokenizer(),
    fixture_tokenizer().digest(),
    identity,
    fixture_inference(),
    11UL,
  )
  let claim = prepared.take_claim()
  let destination = FixedArray::make(8, b'!')
  assert_eq(
    claim
    .push_token_into_status(69, destination, destination_offset=0)
    .copied_bytes(),
    0,
  )
  assert_eq(
    claim
    .push_token_into_status(78, destination, destination_offset=0)
    .copied_bytes(),
    0,
  )
  let matched = claim.push_token_into_status(
    68,
    destination,
    destination_offset=0,
  )
  assert_true(matched.is_ready())
  assert_eq(matched.copied_bytes(), 0)
  assert_true(matched.stop_matched())
  assert_false(
    claim.finish_into_status(destination, destination_offset=0).is_ready(),
  )
}
