///|
fn fixture_inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=16,
    max_new_tokens=12,
    max_context_tokens=32,
    max_token_id=255,
    max_stop_token_ids=4,
    max_stop_strings=3,
    max_stop_string_bytes=16,
    max_trace_bytes=16,
    max_cache_scope_bytes=16,
    max_decoded_delta_bytes=16,
    max_deadline_millis=10000UL,
    max_top_k=256,
    max_temperature=8.0,
  )
}

///|
fn fixture_limits() -> FramedWireLimits raise {
  FramedWireLimits::new(fixture_inference_limits(), max_frame_bytes=4096)
}

///|
fn fixture_identity() -> @spec.ModelIdentity raise {
  @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    @spec.PlanDigest::from_sha256(
      "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
  )
}

///|
fn fixture_request() -> @inference.GenerateRequest raise {
  let limits = fixture_inference_limits()
  @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(42UL),
    model_identity=fixture_identity(),
    input=@inference.Input::from_utf8(b"hello moon", limits),
    max_new_tokens=4,
    context_ceiling=16,
    sampling=@inference.SamplingParameters::sample(
      temperature=0.75,
      top_k=12,
      top_p=0.9,
      limits,
    ),
    sampling_seed=@inference.SamplingSeed::new(99UL),
    stops=@inference.StopConditions::new([2, 7], [b"END", b"stop"], limits),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(5000UL, limits),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"tenant-a", limits),
      ReadWrite,
    ),
    trace=Some(@inference.TraceCorrelation::from_ascii(b"trace-7", limits)),
    limits~,
  )
}

///|
fn frame_bytes(frame : ValidatedRequestFrame) -> Bytes raise {
  let storage = FixedArray::make(frame.length(), b'\x00')
  frame.copy_to(storage, destination_offset=0)
  Bytes::makei(storage.length(), index => storage[index])
}

///|
test "request v1 golden and roundtrip preserve every request field" {
  let limits = fixture_limits()
  let encoded = RequestFrameBuffer::new(limits).encode(fixture_request())
  let bytes = frame_bytes(encoded)
  assert_eq(
    @crypto.bytes_to_hex_string(@crypto.sha256(bytes)),
    "09e534bb3c7d1b9200362b23239dc5c11352ae50811313c800bbe6c717efa72f",
  )
  let source = FixedArray::make(bytes.length(), b'\x00')
  for index, byte in bytes {
    source[index] = byte
  }
  let decoded = RequestFrameBuffer::new(limits)
    .load(source, bytes.length())
    .request()
  assert_eq(decoded.request_id().value(), 42UL)
  assert_eq(decoded.model_identity(), fixture_identity())
  match decoded.input() {
    Text(value) => assert_eq(value.text(), "hello moon")
    _ => fail("expected text input")
  }
  assert_eq(decoded.sampling().mode(), Sample)
  assert_eq(decoded.sampling().top_k(), Some(12))
  assert_eq(decoded.sampling().top_p(), Some(0.9))
  assert_eq(decoded.stops().token_ids().length(), 2)
  assert_eq(decoded.stops().strings()[1], "stop")
  assert_eq(decoded.cache().scope().as_string(), "tenant-a")
  assert_eq(decoded.trace().unwrap().as_string(), "trace-7")
}

///|
test "request token input and greedy canonical form roundtrip" {
  let inference = fixture_inference_limits()
  let request = @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(8UL),
    model_identity=fixture_identity(),
    input=@inference.Input::from_token_ids([1, 2, 3], inference),
    max_new_tokens=2,
    context_ceiling=8,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(3UL),
    stops=@inference.StopConditions::new([], [], inference),
    stream_preference=Buffered,
    deadline=@inference.DeadlineBudget::from_milliseconds(50UL, inference),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"scope", inference),
      Disabled,
    ),
    trace=None,
    limits=inference,
  )
  let owner = RequestFrameBuffer::new(fixture_limits())
  let encoded = owner.encode(request)
  let bytes = frame_bytes(encoded)
  let source = FixedArray::make(bytes.length(), b'\x00')
  for index, byte in bytes {
    source[index] = byte
  }
  match
    RequestFrameBuffer::new(fixture_limits())
    .load(source, bytes.length())
    .request()
    .input() {
    TokenIds(value) => assert_eq(value.token_ids()[2], 3)
    _ => fail("expected token input")
  }
}

///|
fn assert_request_rejected(
  source : FixedArray[Byte],
  length : Int,
) -> Unit raise {
  try RequestFrameBuffer::new(fixture_limits()).load(source, length) catch {
    InvalidFrame(..) => ()
    _ => fail("unexpected request error")
  } noraise {
    _ => fail("expected request rejection")
  }
}

///|
test "request mutations reject magic checksum reserved and semantic flags" {
  let frame = RequestFrameBuffer::new(fixture_limits()).encode(
    fixture_request(),
  )
  let source = FixedArray::make(frame.length(), b'\x00')
  frame.copy_to(source, destination_offset=0)
  source[0] = b'X'
  assert_request_rejected(source, source.length())
  frame.copy_to(source, destination_offset=0)
  source[252] = b'X'
  write_u32(source, 16, frame_checksum(source, source.length()))
  assert_request_rejected(source, source.length())
  frame.copy_to(source, destination_offset=0)
  write_u32(source, 20, read_u32(source, 20) | 0x80)
  write_u32(source, 16, frame_checksum(source, source.length()))
  assert_request_rejected(source, source.length())
  assert_request_rejected(source, source.length() - 1)
}

///|
test "hostile request counts reject before multiplication or allocation" {
  let frame = RequestFrameBuffer::new(fixture_limits()).encode(
    fixture_request(),
  )
  let source = FixedArray::make(frame.length(), b'\x00')
  for offset in [28, 32, 36] {
    frame.copy_to(source, destination_offset=0)
    write_i32(source, offset, 0x7fffffff)
    write_u32(source, 16, frame_checksum(source, source.length()))
    assert_request_rejected(source, source.length())
  }
  frame.copy_to(source, destination_offset=0)
  let first_stop_string_length = REQUEST_HEADER_BYTES + 10 + 8
  write_i32(source, first_stop_string_length, 0x7fffffff)
  write_u32(source, 16, frame_checksum(source, source.length()))
  assert_request_rejected(source, source.length())
}

///|
test "late fixed request evidence rejects before payload ownership" {
  let inference = fixture_inference_limits()
  let request = @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(9UL),
    model_identity=fixture_identity(),
    input=@inference.Input::from_token_ids([1, 2, 3], inference),
    max_new_tokens=2,
    context_ceiling=8,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(3UL),
    stops=@inference.StopConditions::new([], [], inference),
    stream_preference=Buffered,
    deadline=@inference.DeadlineBudget::from_milliseconds(50UL, inference),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"scope", inference),
      Disabled,
    ),
    trace=None,
    limits=inference,
  )
  let frame = RequestFrameBuffer::new(fixture_limits()).encode(request)
  let source = FixedArray::make(frame.length(), b'\x00')
  frame.copy_to(source, destination_offset=0)
  write_u64(source, 88, 0UL)
  write_u32(source, 16, frame_checksum(source, source.length()))
  assert_request_rejected(source, source.length())
  frame.copy_to(source, destination_offset=0)
  write_i32(source, 52, 6)
  write_u32(source, 16, frame_checksum(source, source.length()))
  assert_request_rejected(source, source.length())
}

///|
test "rejected replacement preserves the prior validated request" {
  let owner = RequestFrameBuffer::new(fixture_limits())
  let prior = owner.encode(fixture_request())
  let source = FixedArray::make(prior.length(), b'\x00')
  prior.copy_to(source, destination_offset=0)
  source[0] = b'X'
  try owner.load(source, source.length()) catch {
    InvalidFrame(..) => ()
    _ => fail("unexpected replacement error")
  } noraise {
    _ => fail("expected replacement rejection")
  }
  assert_eq(prior.request().request_id().value(), 42UL)
  ignore(owner.encode(fixture_request()))
  try prior.request() catch {
    StaleFrame => ()
    _ => fail("unexpected stale-frame error")
  } noraise {
    _ => fail("expected replaced frame to become stale")
  }
}

///|
fn event_bytes(frame : ValidatedEventFrame) -> FixedArray[Byte] raise {
  let storage = FixedArray::make(frame.length(), b'\x00')
  frame.copy_to(storage, destination_offset=0)
  storage
}

///|
fn writer_bytes(writer : CanonicalEventWriter) -> FixedArray[Byte] raise {
  let storage = FixedArray::make(writer.length(), b'\x00')
  ignore(writer.copy_to(storage, destination_offset=0))
  storage
}

///|
fn assert_fixed_bytes_equal(
  left : FixedArray[Byte],
  right : FixedArray[Byte],
) -> Unit raise {
  assert_eq(left.length(), right.length())
  for index in 0..<left.length() {
    assert_eq(left[index], right[index])
  }
}

///|
test "direct token writer is byte-identical to canonical event encoding" {
  let inference = fixture_inference_limits()
  let source = FixedArray::make(6, b'!')
  source[1] = b'o'
  source[2] = b'k'
  source[3] = 0xe4
  source[4] = 0xb8
  source[5] = 0xad
  let writer = CanonicalEventWriter::new(fixture_limits())
  writer.prepare_token_bytes(
    token_id=9,
    position=1,
    source,
    source_offset=1,
    length=5,
  )
  let direct = writer_bytes(writer)
  let expected = event_bytes(
    EventFrameBuffer::new(fixture_limits()).encode(
      Token(
        @inference.TokenEvent::new(
          token_id=9,
          decoded_delta=Some(
            @inference.DecodedDelta::from_utf8(b"ok\xe4\xb8\xad", inference),
          ),
          position=1,
          inference,
        ),
      ),
    ),
  )
  assert_fixed_bytes_equal(direct, expected)
  match
    EventFrameBuffer::new(fixture_limits())
    .load(direct, direct.length())
    .event() {
    Token(value) => assert_eq(value.decoded_delta().unwrap().text(), "ok中")
    _ => fail("expected token event")
  }
}

///|
test "direct empty token delta is byte-identical and keeps one credit" {
  let inference = fixture_inference_limits()
  let source = FixedArray::make(0, b'\x00')
  let writer = CanonicalEventWriter::new(fixture_limits())
  writer.prepare_token_bytes(
    token_id=2,
    position=0,
    source,
    source_offset=0,
    length=0,
  )
  let first = writer_bytes(writer)
  let expected = event_bytes(
    EventFrameBuffer::new(fixture_limits()).encode(
      Token(
        @inference.TokenEvent::new(
          token_id=2,
          decoded_delta=None,
          position=0,
          inference,
        ),
      ),
    ),
  )
  assert_fixed_bytes_equal(first, expected)
  assert_false(writer.has_credit())
  try
    writer.prepare_token_bytes(
      token_id=3,
      position=1,
      source,
      source_offset=0,
      length=0,
    )
  catch {
    EventCreditUnavailable => ()
    _ => fail("unexpected credit error")
  } noraise {
    _ => fail("expected busy writer rejection")
  }
  assert_fixed_bytes_equal(writer_bytes(writer), first)
  writer.retire()
  assert_true(writer.has_credit())
  writer.prepare_token_bytes(
    token_id=3,
    position=1,
    source,
    source_offset=0,
    length=0,
  )
  assert_eq(writer.length(), EVENT_HEADER_BYTES)
}

///|
test "direct token rejection is transactional and validates UTF-8" {
  let writer = CanonicalEventWriter::new(fixture_limits())
  let invalid = FixedArray::make(2, b'\x80')
  try
    writer.prepare_token_bytes(
      token_id=1,
      position=0,
      invalid,
      source_offset=0,
      length=2,
    )
  catch {
    InvalidFrame(frame=EventFrame, rule=Payload) => ()
    _ => fail("unexpected UTF-8 error")
  } noraise {
    _ => fail("expected invalid UTF-8 rejection")
  }
  assert_true(writer.has_credit())
  try writer.length() catch {
    StaleFrame => ()
    _ => fail("unexpected empty writer error")
  } noraise {
    _ => fail("expected no published frame")
  }
}

///|
test "all five event variants roundtrip through canonical v1 frames" {
  let inference = fixture_inference_limits()
  let usage = @inference.Usage::initial(
    input_tokens=3,
    cached_input_tokens=1,
    inference,
  ).advance(output_tokens=2, inference)
  let events : Array[@inference.StreamEvent] = [
    Accepted(
      @inference.AcceptedEvent::new(
        model_identity=fixture_identity(),
        effective_limits=@inference.EffectiveLimits::new(
          max_new_tokens=4,
          context_ceiling=16,
          inference,
        ),
      ),
    ),
    Token(
      @inference.TokenEvent::new(
        token_id=9,
        decoded_delta=Some(@inference.DecodedDelta::from_utf8(b"ok", inference)),
        position=1,
        inference,
      ),
    ),
    Usage(usage),
    Completed(
      @inference.CompletedEvent::new(
        finish_reason=TokenLimit,
        final_usage=usage,
      ),
    ),
    Failed(
      @inference.FailedEvent::new(
        code=@inference.PublicErrorCode::from_ascii(b"worker_failed"),
        retryable=true,
      ),
    ),
  ]
  for index, event in events {
    let encoded = EventFrameBuffer::new(fixture_limits()).encode(event)
    let storage = event_bytes(encoded)
    if index == 0 {
      assert_eq(
        @crypto.bytes_to_hex_string(
          @crypto.sha256(Bytes::makei(storage.length(), item => storage[item])),
        ),
        "414795c0b7c87b03846583c0283e00a415c96317641255b6b7996d7b3f4c59fc",
      )
    }
    let decoded = EventFrameBuffer::new(fixture_limits())
      .load(storage, storage.length())
      .event()
    match (index, decoded) {
      (0, Accepted(value)) =>
        assert_eq(value.model_identity(), fixture_identity())
      (1, Token(value)) =>
        assert_eq(value.decoded_delta().unwrap().text(), "ok")
      (2, Usage(value)) => assert_eq(value.total_tokens(), 5)
      (3, Completed(value)) => assert_eq(value.finish_reason(), TokenLimit)
      (4, Failed(value)) => assert_eq(value.code().as_string(), "worker_failed")
      _ => fail("event variant changed")
    }
  }
}

///|
test "event semantic mutation with refreshed checksum is rejected" {
  let inference = fixture_inference_limits()
  let event = @inference.Token(
    @inference.TokenEvent::new(
      token_id=2,
      decoded_delta=None,
      position=0,
      inference,
    ),
  )
  let storage = event_bytes(
    EventFrameBuffer::new(fixture_limits()).encode(event),
  )
  write_u32(storage, 20, 1)
  write_u32(storage, 16, frame_checksum(storage, storage.length()))
  try
    EventFrameBuffer::new(fixture_limits()).load(storage, storage.length())
  catch {
    InvalidFrame(..) => ()
    _ => fail("unexpected event error")
  } noraise {
    _ => fail("expected noncanonical token delta rejection")
  }
}

///|
test "event payload ceilings reject before payload allocation" {
  let inference = fixture_inference_limits()
  let event = @inference.Token(
    @inference.TokenEvent::new(
      token_id=2,
      decoded_delta=None,
      position=0,
      inference,
    ),
  )
  let encoded = event_bytes(
    EventFrameBuffer::new(fixture_limits()).encode(event),
  )
  let oversized = FixedArray::make(EVENT_HEADER_BYTES + 17, b'a')
  for index in 0..<EVENT_HEADER_BYTES {
    oversized[index] = encoded[index]
  }
  write_i32(oversized, 12, oversized.length())
  write_i32(oversized, 56, 17)
  write_u32(oversized, 20, 1)
  write_u32(oversized, 16, frame_checksum(oversized, oversized.length()))
  try
    EventFrameBuffer::new(fixture_limits()).load(oversized, oversized.length())
  catch {
    InvalidFrame(frame=EventFrame, rule=Length) => ()
    _ => fail("unexpected oversized event error")
  } noraise {
    _ => fail("expected oversized event payload rejection")
  }
  let failed = FixedArray::make(EVENT_HEADER_BYTES + 65, b'a')
  for index in 0..<EVENT_HEADER_BYTES {
    failed[index] = encoded[index]
  }
  write_i32(failed, 8, 5)
  write_i32(failed, 12, failed.length())
  write_u32(failed, 20, 0)
  write_i32(failed, 24, 0)
  write_i32(failed, 56, 65)
  write_u32(failed, 16, frame_checksum(failed, failed.length()))
  try
    EventFrameBuffer::new(fixture_limits()).load(failed, failed.length())
  catch {
    InvalidFrame(frame=EventFrame, rule=Length) => ()
    _ => fail("unexpected oversized failed-event error")
  } noraise {
    _ => fail("expected oversized public-code rejection")
  }
}
