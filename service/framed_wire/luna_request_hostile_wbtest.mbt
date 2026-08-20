///|
fn luna_refresh_checksum(source : FixedArray[Byte], length : Int) -> Unit {
  write_u32(source, 16, frame_checksum(source, length))
}

///|
fn luna_assert_frozen_rule(
  source : FixedArray[Byte],
  length : Int,
  expected : FramedWireRule,
) -> Unit raise {
  let mut expected_total = 0UL
  for budget in [1, 17, 65536] {
    let workspace = LunaFramedRequestWorkspace::new(
      fixture_limits(),
      LunaFramedRequestStepBudget::new(budget),
    )
    let work = workspace.begin()
    let mut cursor = 0
    while work.state() != LunaFramedRequestFailed &&
          work.state() != LunaFramedRequestReady {
      match work.state() {
        LunaFramedRequestReceivingPrefix | LunaFramedRequestReceivingBody => {
          if cursor >= length {
            fail("hostile frame remained incomplete")
          }
          let consumed = work.offer(
            source,
            source_offset=cursor,
            length=length - cursor,
          )
          assert_true(consumed > 0)
          assert_luna_work_charge(work, budget)
          cursor += consumed
        }
        LunaFramedRequestValidating => luna_progress_once(work, budget)
        LunaFramedRequestReady | LunaFramedRequestFailed => ()
      }
    }
    assert_eq(work.state(), LunaFramedRequestFailed)
    match work.failure() {
      InvalidFrame(frame=RequestFrame, rule=actual) =>
        assert_eq(actual, expected)
      _ => fail("hostile frame produced a noncanonical error")
    }
    let total = work.total_work_units()
    if expected_total == 0UL {
      expected_total = total
    } else {
      assert_eq(total, expected_total)
    }
    work.abort()
  }
  try RequestFrameBuffer::new(fixture_limits()).load(source, length) catch {
    InvalidFrame(frame=RequestFrame, rule=actual) => assert_eq(actual, expected)
    _ => fail("compatibility facade produced a noncanonical error")
  } noraise {
    _ => fail("hostile frame passed compatibility validation")
  }
}

///|
fn luna_equal_stop_source() -> (FixedArray[Byte], Int) raise {
  let limits = fixture_inference_limits()
  let request = @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(43UL),
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
    stops=@inference.StopConditions::new([2, 7], [b"same", b"stop"], limits),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(5000UL, limits),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"tenant-a", limits),
      ReadWrite,
    ),
    trace=Some(@inference.TraceCorrelation::from_ascii(b"trace-7", limits)),
    limits~,
  )
  let frame = RequestFrameBuffer::new(fixture_limits()).encode(request)
  let source = FixedArray::make(frame.length(), b'\x00')
  frame.copy_to(source, destination_offset=0)
  (source, source.length())
}

///|
test "Luna framed request rejects frozen header checksum and layout rules" {
  let (source, length) = luna_request_source()
  source[0] = b'X'
  luna_assert_frozen_rule(source, length, Magic)

  let (source, length) = luna_request_source()
  source[16] = source[16] ^ b'\x01'
  luna_assert_frozen_rule(source, length, Checksum)

  let (source, length) = luna_request_source()
  source[252] = b'X'
  luna_refresh_checksum(source, length)
  luna_assert_frozen_rule(source, length, Reserved)

  let (source, length) = luna_request_source()
  write_i32(source, 64, 257)
  luna_refresh_checksum(source, length)
  luna_assert_frozen_rule(source, length, Scalar)

  let (source, length) = luna_request_source()
  source[112] = b'g'
  write_i32(source, 52, 0)
  luna_refresh_checksum(source, length)
  luna_assert_frozen_rule(source, length, Identity)

  let (source, length) = luna_request_source()
  write_i32(source, 274, 0x7fffffff)
  luna_refresh_checksum(source, length)
  luna_assert_frozen_rule(source, length, StopTable)

  let (source, length) = luna_request_source()
  write_u32(source, 20, read_u32(source, 20) & 0xffffffdf)
  luna_refresh_checksum(source, length)
  luna_assert_frozen_rule(source, length, CanonicalEncoding)
}

///|
test "Luna framed request rejects hostile UTF-8 at exact field boundaries" {
  let (source, length) = luna_request_source()
  source[256] = b'\xc0'
  source[257] = b'\x80'
  luna_refresh_checksum(source, length)
  luna_assert_frozen_rule(source, length, Input)

  let (source, length) = luna_request_source()
  source[256] = b'\xed'
  source[257] = b'\xa0'
  source[258] = b'\x80'
  luna_refresh_checksum(source, length)
  luna_assert_frozen_rule(source, length, Input)

  let (source, length) = luna_request_source()
  source[265] = b'\xe2'
  luna_refresh_checksum(source, length)
  luna_assert_frozen_rule(source, length, Input)

  let (source, length) = luna_equal_stop_source()
  source[278] = b'\xed'
  source[279] = b'\xa0'
  source[280] = b'\x80'
  luna_refresh_checksum(source, length)
  luna_assert_frozen_rule(source, length, StopTable)
}

///|
test "Luna framed request rejects duplicate stop tokens and strings" {
  let (source, length) = luna_request_source()
  write_i32(source, 270, 2)
  luna_refresh_checksum(source, length)
  luna_assert_frozen_rule(source, length, StopTable)

  let (source, length) = luna_equal_stop_source()
  for index in 0..<4 {
    source[286 + index] = source[278 + index]
  }
  luna_refresh_checksum(source, length)
  luna_assert_frozen_rule(source, length, StopTable)
}

///|
test "compatibility load rejects trailing bytes and remains reusable" {
  let (source, length) = luna_request_source()
  let trailing = FixedArray::make(length + 1, b'X')
  for index in 0..<length {
    trailing[index] = source[index]
  }
  let owner = RequestFrameBuffer::new(fixture_limits())
  try owner.load(trailing, length + 1) catch {
    InvalidFrame(frame=RequestFrame, rule=Length) => ()
    _ => fail("trailing frame produced a noncanonical error")
  } noraise {
    _ => fail("trailing frame was accepted")
  }
  assert_eq(owner.load(source, length).request().request_id().value(), 42UL)
}

///|
test "Luna framed request preserves frozen truncated-layout error" {
  let (source, _) = luna_request_source()
  let length = 260
  write_u32(source, 12, length.reinterpret_as_uint())
  write_i32(source, 244, length - REQUEST_HEADER_BYTES)
  luna_refresh_checksum(source, length)
  for budget in [1, 17, 65536] {
    let workspace = LunaFramedRequestWorkspace::new(
      fixture_limits(),
      LunaFramedRequestStepBudget::new(budget),
    )
    let work = workspace.begin()
    let mut cursor = 0
    while work.state() != LunaFramedRequestFailed {
      match work.state() {
        LunaFramedRequestReceivingPrefix | LunaFramedRequestReceivingBody => {
          let consumed = work.offer(
            source,
            source_offset=cursor,
            length=length - cursor,
          )
          cursor += consumed
        }
        LunaFramedRequestValidating => ignore(work.progress())
        LunaFramedRequestReady => fail("truncated request became ready")
        LunaFramedRequestFailed => ()
      }
    }
    match work.failure() {
      FrameTooLarge(frame=RequestFrame) => ()
      _ => fail("truncated layout lost its frozen error")
    }
    work.abort()
  }
  try RequestFrameBuffer::new(fixture_limits()).load(source, length) catch {
    FrameTooLarge(frame=RequestFrame) => ()
    _ => fail("compatibility truncated layout lost its frozen error")
  } noraise {
    _ => fail("truncated layout was accepted")
  }
}
