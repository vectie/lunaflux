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
fn session_received(
  stream : @inference.StreamPreference,
) -> @request_admission.ReceivedRequest raise {
  let limits = inference_limits()
  let request = @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(901UL),
    model_identity=identity(),
    input=@inference.Input::from_token_ids([1, 2], limits),
    max_new_tokens=2,
    context_ceiling=4,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(9UL),
    stops=@inference.StopConditions::new([], [], limits),
    stream_preference=stream,
    deadline=@inference.DeadlineBudget::from_milliseconds(1000UL, limits),
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

  let model = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let kernel = @approved_fs.ApprovedRoot::open_absolute("/private/tmp")
  let wire = @framed_wire.FramedWireLimits::new(
    inference_limits(),
    max_frame_bytes=4096,
  )
  let received = session_received(Streaming)
  let blueprint = scheduler_blueprint()
  let prepared = @online_session.prepare_owned_session(
    received,
    tokenizer,
    tokenizer.digest(),
    identity(),
    inference_limits(),
    wire,
    blueprint,
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
  let session = prepared.take_ready()
  assert_true(session.has_event())
  let bytes = FixedArray::make(session.event_length(), b'\x00')
  assert_eq(session.copy_event_to(bytes, destination_offset=0), bytes.length())
  match
    @framed_wire.EventFrameBuffer::new(wire).load(bytes, bytes.length()).event() {
    Accepted(value) => assert_eq(value.model_identity(), identity())
    _ => raise Failure("online session did not publish Accepted")
  }
  session.ack_event()
  assert_false(session.has_event())
  try session.event_length() catch {
    SessionFailed(Invalid(EventCredit)) => ()
    _ => raise Failure("unexpected stale online event error")
  } noraise {
    _ => raise Failure("stale online event remained usable")
  }
  session.begin_abort()
  assert_eq(session.progress_cleanup(), OnlineSessionCleanupPending)
  assert_eq(session.progress_cleanup(), OnlineSessionCleanupClosed)
}
