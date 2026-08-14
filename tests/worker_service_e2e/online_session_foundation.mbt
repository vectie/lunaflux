///|
fn session_tokenizer() -> @tokenizer.TokenizerSpec raise {
  let vocabulary : Array[@tokenizer.VocabularyEntry] = []
  for value in 0..<256 {
    vocabulary.push({ token_id: value, piece: [value.to_byte()], kind: Normal })
  }
  @tokenizer.TokenizerSpec::new(
    vocabulary,
    [],
    @tokenizer.TokenizerDigest::parse("0123456789abcdef".repeat(4)),
    {
      max_input_bytes: 4,
      max_output_tokens: 4,
      max_decoded_bytes: 8,
      max_vocab_entries: 256,
      max_merge_rules: 1,
      max_token_bytes: 1,
      max_special_tokens: 1,
    },
  )
}

///|
fn invalid_output_session_tokenizer() -> @tokenizer.TokenizerSpec raise {
  let vocabulary : Array[@tokenizer.VocabularyEntry] = []
  for value in 0..<256 {
    let piece = if value == 0 {
      b"\xc0"
    } else if value == 0xc0 {
      b"\x00"
    } else {
      Bytes::make(1, value.to_byte())
    }
    vocabulary.push({ token_id: value, piece, kind: Normal })
  }
  @tokenizer.TokenizerSpec::new(
    vocabulary,
    [],
    @tokenizer.TokenizerDigest::parse("abcdef0123456789".repeat(4)),
    {
      max_input_bytes: 4,
      max_output_tokens: 4,
      max_decoded_bytes: 8,
      max_vocab_entries: 256,
      max_merge_rules: 1,
      max_token_bytes: 1,
      max_special_tokens: 1,
    },
  )
}

///|
fn string_session_tokenizer(
  piece : Bytes,
  digest : String,
) -> @tokenizer.TokenizerSpec raise {
  let vocabulary : Array[@tokenizer.VocabularyEntry] = []
  let displaced = if piece.length() == 1 { piece[0].to_int() } else { -1 }
  vocabulary.push({ token_id: 0, piece, kind: Normal })
  for value in 1..<256 {
    let byte_piece = if value == displaced {
      b"\x00"
    } else {
      [value.to_byte()]
    }
    vocabulary.push({ token_id: value, piece: byte_piece, kind: Normal })
  }
  if displaced < 0 {
    vocabulary.push({ token_id: 256, piece: b"\x00", kind: Normal })
  }
  @tokenizer.TokenizerSpec::new(
    vocabulary,
    [],
    @tokenizer.TokenizerDigest::parse(digest),
    {
      max_input_bytes: 4,
      max_output_tokens: 4,
      max_decoded_bytes: 8,
      max_vocab_entries: 257,
      max_merge_rules: 1,
      max_token_bytes: 2,
      max_special_tokens: 1,
    },
  )
}

///|
fn session_received(
  stream : @inference.StreamPreference,
  request_id? : UInt64 = 901UL,
  stop_token? : Int = -1,
  stop_string? : Bytes = b"",
  max_new_tokens? : Int = 2,
  deadline_millis? : UInt64 = 1000UL,
) -> @request_admission.ReceivedRequest raise {
  let limits = inference_limits()
  let stop_tokens : Array[Int] = []
  if stop_token >= 0 {
    stop_tokens.push(stop_token)
  }
  let stop_strings : Array[Bytes] = []
  if stop_string.length() > 0 {
    stop_strings.push(stop_string)
  }
  let request = @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(request_id),
    model_identity=identity(),
    input=@inference.Input::from_token_ids([1, 2], limits),
    max_new_tokens~,
    context_ceiling=4,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(9UL),
    stops=@inference.StopConditions::new(stop_tokens, stop_strings, limits),
    stream_preference=stream,
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
  let wire = @framed_wire.FramedWireLimits::new(limits, max_frame_bytes=4096)
  let frame = @framed_wire.RequestFrameBuffer::new(wire).encode(request)
  let source = FixedArray::make(frame.length(), b'\x00')
  frame.copy_to(source, destination_offset=0)
  @request_admission.receive(
    @framed_wire.RequestFrameBuffer::new(wire),
    source,
    source.length(),
    @monotonic_clock.MonotonicClock::system(),
  )
}

///|
fn prepare_foundation_session(
  executable : Bytes,
  tokenizer : @tokenizer.TokenizerSpec,
  received : @request_admission.ReceivedRequest,
) -> @online_session.OnlineSession raise {
  let model = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let prepared = @online_session.prepare_owned_session(
    received,
    tokenizer,
    tokenizer.digest(),
    identity(),
    inference_limits(),
    @framed_wire.FramedWireLimits::new(inference_limits(), max_frame_bytes=4096),
    scheduler_blueprint(),
    service_binding(),
    executable,
    startup(),
    bootstrap_source(),
    process_limits(),
    model,
    kernel,
  )
  model.close()
  kernel.close()
  assert_eq(prepared.state(), OnlineSessionReady)
  prepared.take_ready()
}

///|
fn foundation_event(
  session : @online_session.OnlineSession,
) -> @inference.StreamEvent raise {
  let wire = @framed_wire.FramedWireLimits::new(
    inference_limits(),
    max_frame_bytes=4096,
  )
  let bytes = FixedArray::make(session.event_length(), b'\x00')
  assert_eq(session.copy_event_to(bytes, destination_offset=0), bytes.length())
  @framed_wire.EventFrameBuffer::new(wire).load(bytes, bytes.length()).event()
}

///|
fn progress_foundation_to_event(
  session : @online_session.OnlineSession,
) -> Unit raise {
  let mut attempts = 0
  while !session.has_event() && attempts < 1000000 {
    match session.progress() {
      OnlineSessionTerminalizationRequired =>
        ignore(session.progress_terminalization())
      OnlineSessionPending | OnlineSessionEventReady => ()
    }
    attempts += 1
  }
  if !session.has_event() {
    raise Failure("online session did not publish bounded progress")
  }
}

///|
fn progress_foundation_to_terminalization(
  session : @online_session.OnlineSession,
) -> Unit raise {
  let mut attempts = 0
  while attempts < 1000000 {
    match session.progress() {
      OnlineSessionTerminalizationRequired => return
      OnlineSessionPending => ()
      OnlineSessionEventReady =>
        raise Failure("online failure unexpectedly published on reactor")
    }
    attempts += 1
  }
  raise Failure("online failure did not require bounded terminalization")
}

///|
fn run_online_session_foundation(executable : Bytes) -> Unit raise {
  let tokenizer = session_tokenizer()
  let narrow_wire = @framed_wire.FramedWireLimits::new(
    inference_limits(),
    max_frame_bytes=256,
  )
  let narrow_model = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let narrow_kernel = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  try
    @online_session.prepare_owned_session(
      session_received(Streaming),
      tokenizer,
      tokenizer.digest(),
      identity(),
      inference_limits(),
      narrow_wire,
      scheduler_blueprint(),
      service_binding(),
      executable,
      startup(),
      bootstrap_source(),
      process_limits(),
      narrow_model,
      narrow_kernel,
    )
  catch {
    @online_session.PreparationFailed(WireFailure) => ()
    _ => raise Failure("unexpected online wire-capacity preflight error")
  } noraise {
    _ => raise Failure("undersized online event storage reached rooted startup")
  }
  narrow_model.close()
  narrow_kernel.close()

  let session = prepare_foundation_session(
    executable,
    tokenizer,
    session_received(Streaming),
  )
  assert_true(session.has_event())
  match foundation_event(session) {
    Accepted(value) => assert_eq(value.model_identity(), identity())
    _ => raise Failure("online session did not publish Accepted")
  }
  try session.progress() catch {
    SessionFailed(Invalid(EventCredit)) => ()
    _ => raise Failure("unexpected pinned online progress error")
  } noraise {
    _ => raise Failure("pinned Accepted allowed online mutation")
  }
  session.ack_event()
  assert_false(session.has_event())
  try session.event_length() catch {
    SessionFailed(Invalid(EventCredit)) => ()
    _ => raise Failure("unexpected stale online event error")
  } noraise {
    _ => raise Failure("stale online event remained usable")
  }

  for position in 0..<2 {
    progress_foundation_to_event(session)
    match foundation_event(session) {
      Token(value) => {
        assert_eq(value.token_id(), 0)
        assert_eq(value.position(), position)
      }
      _ => raise Failure("online maximum path did not publish Token")
    }
    session.ack_event()
  }
  progress_foundation_to_event(session)
  match foundation_event(session) {
    Usage(value) => {
      assert_eq(value.input_tokens(), 2)
      assert_eq(value.output_tokens(), 2)
    }
    _ => raise Failure("online maximum path did not publish Usage")
  }
  session.ack_event()
  match foundation_event(session) {
    Completed(value) => {
      assert_eq(value.finish_reason(), TokenLimit)
      assert_eq(value.final_usage().output_tokens(), 2)
      assert_true(value.final_delta() is None)
    }
    _ => raise Failure("online maximum path did not publish Completed")
  }
  session.ack_event()
  assert_eq(session.progress_cleanup(), OnlineSessionCleanupClosed)

  let stopped = prepare_foundation_session(
    executable,
    tokenizer,
    session_received(Streaming, request_id=902UL, stop_token=0),
  )
  stopped.ack_event()
  progress_foundation_to_event(stopped)
  match foundation_event(stopped) {
    Usage(value) => assert_eq(value.output_tokens(), 1)
    _ => raise Failure("online stop-token path leaked its physical token")
  }
  stopped.ack_event()
  match foundation_event(stopped) {
    Completed(value) => {
      assert_eq(value.finish_reason(), StopSequence)
      assert_eq(value.final_usage().output_tokens(), 1)
      assert_true(value.final_delta() is None)
    }
    _ => raise Failure("online stop-token path did not publish Completed")
  }
  stopped.ack_event()
  assert_eq(stopped.progress_cleanup(), OnlineSessionCleanupClosed)

  let invalid_output = prepare_foundation_session(
    executable,
    invalid_output_session_tokenizer(),
    session_received(Streaming, request_id=905UL),
  )
  invalid_output.ack_event()
  progress_foundation_to_terminalization(invalid_output)
  assert_eq(invalid_output.progress(), OnlineSessionTerminalizationRequired)
  progress_foundation_to_event(invalid_output)
  match foundation_event(invalid_output) {
    Usage(value) => assert_eq(value.output_tokens(), 1)
    _ => raise Failure("invalid output did not publish Usage")
  }
  invalid_output.request_cancel()
  invalid_output.check_deadline()
  invalid_output.ack_event()
  match foundation_event(invalid_output) {
    Failed(value) => {
      assert_eq(value.code().as_string(), "output_invalid")
      assert_false(value.retryable())
    }
    _ => raise Failure("invalid output did not publish canonical Failed")
  }
  invalid_output.ack_event()
  assert_eq(invalid_output.progress_cleanup(), OnlineSessionCleanupClosed)

  let aborting = prepare_foundation_session(
    executable,
    tokenizer,
    session_received(Streaming, request_id=903UL),
  )
  aborting.ack_event()
  aborting.begin_abort()
  assert_eq(aborting.progress_cleanup(), OnlineSessionCleanupPending)
  assert_eq(aborting.progress_cleanup(), OnlineSessionCleanupClosed)

  let string_zero_tokenizer = string_session_tokenizer(b"x", "1".repeat(64))
  let string_zero_received = session_received(
    Streaming,
    request_id=904UL,
    stop_string=b"x",
    max_new_tokens=2,
  )
  assert_string_stop_session(
    executable,
    string_zero_tokenizer,
    string_zero_received,
    [],
    1,
  )
  assert_string_stop_session(
    executable,
    string_session_tokenizer(b"ax", "2".repeat(64)),
    session_received(
      Streaming,
      request_id=907UL,
      stop_string=b"x",
      max_new_tokens=2,
    ),
    [Some("a")],
    1,
  )
  assert_string_stop_session(
    executable,
    string_session_tokenizer(b"a", "3".repeat(64)),
    session_received(
      Streaming,
      request_id=908UL,
      stop_string=b"aa",
      max_new_tokens=2,
    ),
    [None],
    2,
  )
  assert_string_stop_session(
    executable,
    string_session_tokenizer(b"ax", "4".repeat(64)),
    session_received(
      Streaming,
      request_id=909UL,
      stop_string=b"x",
      max_new_tokens=1,
    ),
    [Some("a")],
    1,
  )
  assert_string_cut_abort(executable, 910UL, false)
  assert_string_cut_abort(executable, 911UL, true)
}
