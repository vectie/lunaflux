///|
fn luna_text_expected_request() -> @inference.GenerateRequest raise {
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
    stops=@inference.StopConditions::new([], [], limits),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(5000UL, limits),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"tenant-a", limits),
      Disabled,
    ),
    trace=None,
    limits~,
  )
}

///|
fn luna_text_encode(budget : Int) -> (FixedArray[Byte], UInt64) raise {
  let limits = fixture_limits()
  let inference = fixture_inference_limits()
  let workspace = LunaFramedTextRequestWorkspace::new(
    limits,
    LunaFramedTextRequestStepBudget::new(budget),
    fixture_identity(),
    @inference.CacheScope::from_ascii(b"tenant-a", inference),
  )
  let input = b"hello moon"
  let write = workspace.begin_text(
    byte_count=input.length(),
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(42UL),
    max_new_tokens=4,
    context_ceiling=16,
    sampling=@inference.SamplingParameters::sample(
      temperature=0.75,
      top_k=12,
      top_p=0.9,
      inference,
    ),
    sampling_seed=@inference.SamplingSeed::new(99UL),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(5000UL, inference),
  )
  for byte in input {
    write.push_byte(byte)
  }
  let work = write.finish()
  while work.state() == LunaFramedTextRequestPending {
    assert_eq(work.progress(), LunaFramedTextRequestAdvanced)
    assert_true(work.last_work_units() > 0)
    assert_true(work.last_work_units() <= budget)
  }
  let total = work.total_work_units()
  let view = work.take_view()
  let result : FixedArray[Byte] = FixedArray::make(view.length(), 0)
  let mut offset = 0
  while offset < result.length() {
    let length = Int::min(budget, result.length() - offset)
    ignore(
      view.copy_chunk_to(
        result,
        destination_offset=offset,
        source_offset=offset,
        length~,
      ),
    )
    offset += length
  }
  view.release()
  (result, total)
}

///|
test "Luna framed text writer is byte-identical to RequestFrameBuffer v1" {
  let expected_frame = RequestFrameBuffer::new(fixture_limits()).encode(
    luna_text_expected_request(),
  )
  let expected : FixedArray[Byte] = FixedArray::make(expected_frame.length(), 0)
  expected_frame.copy_to(expected, destination_offset=0)
  let mut baseline_total = 0UL
  for budget in [1, 17, 65536] {
    let (actual, total) = luna_text_encode(budget)
    assert_eq(actual.length(), expected.length())
    for index in 0..<actual.length() {
      assert_eq(actual[index], expected[index])
    }
    if baseline_total == 0UL {
      baseline_total = total
    } else {
      assert_eq(total, baseline_total)
    }
    let decoded = RequestFrameBuffer::new(fixture_limits())
      .load(actual, actual.length())
      .request()
    assert_eq(decoded.request_id().value(), 42UL)
    assert_eq(decoded.cache().permission(), Disabled)
    assert_eq(decoded.cache().scope().as_string(), "tenant-a")
  }
}

///|
test "Luna framed text writer abort reuse and stale aliases are exact" {
  let inference = fixture_inference_limits()
  let workspace = LunaFramedTextRequestWorkspace::new(
    fixture_limits(),
    LunaFramedTextRequestStepBudget::new(1),
    fixture_identity(),
    @inference.CacheScope::from_ascii(b"tenant-a", inference),
  )
  let write = workspace.begin_text(
    byte_count=1,
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(1UL),
    max_new_tokens=1,
    context_ceiling=2,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(1UL),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(10UL, inference),
  )
  write.abort()
  try write.push_byte(b'x') catch {
    LunaFramedTextRequestStale => ()
    _ => fail("wrong stale error")
  } noraise {
    _ => fail("stale writer remained live")
  }
  let reused = workspace.begin_text(
    byte_count=1,
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(2UL),
    max_new_tokens=1,
    context_ceiling=2,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(2UL),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(10UL, inference),
  )
  reused.push_byte(b'x')
  reused.finish().abort()
}
