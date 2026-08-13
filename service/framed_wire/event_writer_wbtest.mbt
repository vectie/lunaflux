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
test "direct scalar event writers are byte-identical to canonical v2" {
  let inference = fixture_inference_limits()
  let effective = @inference.EffectiveLimits::new(
    max_new_tokens=4,
    context_ceiling=16,
    inference,
  )
  let writer = CanonicalEventWriter::new(fixture_limits())
  writer.prepare_accepted(fixture_identity(), effective)
  assert_fixed_bytes_equal(
    writer_bytes(writer),
    event_bytes(
      EventFrameBuffer::new(fixture_limits()).encode(
        Accepted(
          @inference.AcceptedEvent::new(
            model_identity=fixture_identity(),
            effective_limits=effective,
          ),
        ),
      ),
    ),
  )
  writer.retire()
  writer.prepare_usage(input_tokens=3, cached_input_tokens=1, output_tokens=2)
  let usage = @inference.Usage::initial(
    input_tokens=3,
    cached_input_tokens=1,
    inference,
  ).advance(output_tokens=2, inference)
  assert_fixed_bytes_equal(
    writer_bytes(writer),
    event_bytes(EventFrameBuffer::new(fixture_limits()).encode(Usage(usage))),
  )
  writer.retire()
  let tail : FixedArray[Byte] = [b't', b'a', b'i', b'l']
  writer.prepare_completed(
    TokenLimit,
    input_tokens=3,
    cached_input_tokens=1,
    output_tokens=2,
    tail,
    tail_offset=0,
    tail_length=4,
  )
  assert_fixed_bytes_equal(
    writer_bytes(writer),
    event_bytes(
      EventFrameBuffer::new(fixture_limits()).encode(
        Completed(
          @inference.CompletedEvent::with_final_delta(
            finish_reason=TokenLimit,
            final_usage=usage,
            final_delta=@inference.DecodedDelta::from_utf8(b"tail", inference),
            inference,
          ),
        ),
      ),
    ),
  )
  writer.retire()
  let code : FixedArray[Byte] = [
    b'o', b'u', b't', b'p', b'u', b't', b'_', b'i', b'n', b'v', b'a', b'l', b'i',
    b'd',
  ]
  writer.prepare_failed(
    code,
    code_offset=0,
    code_length=code.length(),
    retryable=false,
  )
  assert_fixed_bytes_equal(
    writer_bytes(writer),
    event_bytes(
      EventFrameBuffer::new(fixture_limits()).encode(
        Failed(
          @inference.FailedEvent::new(
            code=@inference.PublicErrorCode::from_ascii(b"output_invalid"),
            retryable=false,
          ),
        ),
      ),
    ),
  )
}

///|
test "completed v2 tail flags UTF-8 and bounds are strict and transactional" {
  let writer = CanonicalEventWriter::new(fixture_limits())
  let invalid : FixedArray[Byte] = [b'\x80']
  try
    writer.prepare_completed(
      StopSequence,
      input_tokens=1,
      cached_input_tokens=0,
      output_tokens=1,
      invalid,
      tail_offset=0,
      tail_length=1,
    )
  catch {
    InvalidFrame(frame=EventFrame, rule=Payload) => ()
    _ => fail("unexpected completed tail error")
  } noraise {
    _ => fail("expected invalid completed tail rejection")
  }
  assert_true(writer.has_credit())
  let oversized = FixedArray::make(
    fixture_inference_limits().max_decoded_delta_bytes() + 1,
    b'a',
  )
  try
    writer.prepare_completed(
      StopSequence,
      input_tokens=1,
      cached_input_tokens=0,
      output_tokens=1,
      oversized,
      tail_offset=0,
      tail_length=oversized.length(),
    )
  catch {
    InvalidFrame(frame=EventFrame, rule=Length) => ()
    _ => fail("unexpected completed tail bound error")
  } noraise {
    _ => fail("expected oversized completed tail rejection")
  }
  assert_true(writer.has_credit())
  let empty = FixedArray::make(0, b'\x00')
  writer.prepare_completed(
    Cancelled,
    input_tokens=1,
    cached_input_tokens=0,
    output_tokens=0,
    empty,
    tail_offset=0,
    tail_length=0,
  )
  let storage = writer_bytes(writer)
  assert_eq(read_u32(storage, 20), 0)
  assert_eq(storage.length(), EVENT_HEADER_BYTES)
  write_u32(storage, 20, 1)
  write_u32(storage, 16, frame_checksum(storage, storage.length()))
  try
    EventFrameBuffer::new(fixture_limits()).load(storage, storage.length())
  catch {
    InvalidFrame(frame=EventFrame, rule=CanonicalEncoding) => ()
    _ => fail("unexpected completed flag error")
  } noraise {
    _ => fail("expected completed flag mismatch")
  }
}

///|
test "event v2 rejects the retired v1 version and nonzero reserved bytes" {
  let storage = event_bytes(
    EventFrameBuffer::new(fixture_limits()).encode(
      Usage(
        @inference.Usage::initial(
          input_tokens=1,
          cached_input_tokens=0,
          fixture_inference_limits(),
        ),
      ),
    ),
  )
  write_u32(storage, 4, 1)
  write_u32(storage, 16, frame_checksum(storage, storage.length()))
  try
    EventFrameBuffer::new(fixture_limits()).load(storage, storage.length())
  catch {
    InvalidFrame(frame=EventFrame, rule=Version) => ()
    _ => fail("unexpected retired event version error")
  } noraise {
    _ => fail("expected retired event version rejection")
  }
  write_u32(storage, 4, event_version)
  storage[200] = 1
  write_u32(storage, 16, frame_checksum(storage, storage.length()))
  try
    EventFrameBuffer::new(fixture_limits()).load(storage, storage.length())
  catch {
    InvalidFrame(frame=EventFrame, rule=Reserved) => ()
    _ => fail("unexpected reserved event error")
  } noraise {
    _ => fail("expected reserved event rejection")
  }
}

///|
test "direct scalar writers reject hostile envelopes transactionally" {
  let writer = CanonicalEventWriter::new(fixture_limits())
  let bytes : FixedArray[Byte] = [b'o', b'k']
  for offset in [-1, 3] {
    try
      writer.prepare_completed(
        Cancelled,
        input_tokens=1,
        cached_input_tokens=0,
        output_tokens=0,
        bytes,
        tail_offset=offset,
        tail_length=1,
      )
    catch {
      InvalidFrame(frame=EventFrame, rule=Length) => ()
      _ => fail("unexpected hostile tail offset error")
    } noraise {
      _ => fail("expected hostile tail offset rejection")
    }
    assert_true(writer.has_credit())
  }
  let invalid_code : FixedArray[Byte] = [b'b', b'a', b'd', b'/']
  for offset in [-1, invalid_code.length() + 1] {
    try
      writer.prepare_failed(
        invalid_code,
        code_offset=offset,
        code_length=1,
        retryable=false,
      )
    catch {
      InvalidFrame(frame=EventFrame, rule=Payload) => ()
      _ => fail("unexpected hostile failure offset error")
    } noraise {
      _ => fail("expected hostile failure offset rejection")
    }
    assert_true(writer.has_credit())
  }
  for
    code in [
      FixedArray::make(0, b'a'),
      FixedArray::make(65, b'a'),
      invalid_code,
    ] {
    try
      writer.prepare_failed(
        code,
        code_offset=0,
        code_length=code.length(),
        retryable=false,
      )
    catch {
      InvalidFrame(frame=EventFrame, rule=Payload) => ()
      _ => fail("unexpected hostile failure code error")
    } noraise {
      _ => fail("expected hostile failure code rejection")
    }
    assert_true(writer.has_credit())
  }
  for usage_case in [(-1, 0, 0), (1, 2, 0), (32, 0, 1), (1, 0, 13)] {
    let (input, cached, output) = usage_case
    try
      writer.prepare_usage(
        input_tokens=input,
        cached_input_tokens=cached,
        output_tokens=output,
      )
    catch {
      InvalidFrame(frame=EventFrame, rule=Scalar) => ()
      _ => fail("unexpected hostile usage error")
    } noraise {
      _ => fail("expected hostile usage rejection")
    }
    assert_true(writer.has_credit())
  }
  let broad = @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=32,
    max_new_tokens=24,
    max_context_tokens=64,
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
  try
    writer.prepare_accepted(
      fixture_identity(),
      @inference.EffectiveLimits::new(
        max_new_tokens=20,
        context_ceiling=40,
        broad,
      ),
    )
  catch {
    InvalidFrame(frame=EventFrame, rule=Scalar) => ()
    _ => fail("unexpected cross-envelope accepted error")
  } noraise {
    _ => fail("expected cross-envelope accepted rejection")
  }
  assert_true(writer.has_credit())
  writer.prepare_usage(input_tokens=1, cached_input_tokens=0, output_tokens=0)
  let pinned = writer_bytes(writer)
  try
    writer.prepare_failed(
      invalid_code,
      code_offset=0,
      code_length=invalid_code.length(),
      retryable=false,
    )
  catch {
    EventCreditUnavailable => ()
    _ => fail("unexpected pinned credit error")
  } noraise {
    _ => fail("expected pinned frame rejection")
  }
  assert_false(writer.has_credit())
  assert_fixed_bytes_equal(writer_bytes(writer), pinned)
}

///|
