///|
fn luna_sampling_expected_request() -> @inference.GenerateRequest raise {
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
      Disabled,
    ),
    trace=None,
    limits~,
  )
}

///|
fn luna_sampling_scalar_frame(budget : Int) -> (FixedArray[Byte], UInt64) raise {
  let inference = fixture_inference_limits()
  let workspace = LunaFramedTextRequestWorkspace::new(
    fixture_limits(),
    LunaFramedTextRequestStepBudget::new(budget),
    fixture_identity(),
    @inference.CacheScope::from_ascii(b"tenant-a", inference),
  )
  let input = b"hello moon"
  let write = workspace.begin_text_with_sampling_scalars(
    byte_count=input.length(),
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(42UL),
    max_new_tokens=4,
    context_ceiling=16,
    sampling_mode=Sample,
    temperature=0.75,
    has_top_k=true,
    top_k=12,
    has_top_p=true,
    top_p=0.9,
    sampling_seed_value=99UL,
    stop_token_count=2,
    stop_string_count=2,
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(5000UL, inference),
  )
  for byte in input {
    write.push_byte(byte)
  }
  write.push_stop_token(2)
  write.push_stop_token(7)
  for value in [b"END", b"stop"] {
    write.push_stop_string_length(value.length())
    for byte in value {
      write.push_stop_string_byte(byte)
    }
  }
  let work = write.finish()
  while work.state() == LunaFramedTextRequestPending {
    ignore(work.progress())
  }
  let total = work.total_work_units()
  let view = work.take_view()
  let result : FixedArray[Byte] = FixedArray::make(view.length(), 0)
  let mut cursor = 0
  while cursor < result.length() {
    let amount = Int::min(budget, result.length() - cursor)
    ignore(
      view.copy_chunk_to(
        result,
        destination_offset=cursor,
        source_offset=cursor,
        length=amount,
      ),
    )
    cursor += amount
  }
  view.release()
  (result, total)
}

///|
test "Luna scalar sampling writer preserves canonical stops and exact budgets" {
  let expected_frame = RequestFrameBuffer::new(fixture_limits()).encode(
    luna_sampling_expected_request(),
  )
  let expected : FixedArray[Byte] = FixedArray::make(expected_frame.length(), 0)
  expected_frame.copy_to(expected, destination_offset=0)
  let mut expected_total = 0UL
  for budget in [1, 17, 65536] {
    let (actual, total) = luna_sampling_scalar_frame(budget)
    assert_eq(actual.length(), expected.length())
    for index in 0..<actual.length() {
      assert_eq(actual[index], expected[index])
    }
    if expected_total == 0UL {
      expected_total = total
    } else {
      assert_eq(total, expected_total)
    }
  }
}

///|
test "Luna scalar sampling writer rejects incomplete and oversized stops" {
  let inference = fixture_inference_limits()
  let workspace = LunaFramedTextRequestWorkspace::new(
    fixture_limits(),
    LunaFramedTextRequestStepBudget::new(1),
    fixture_identity(),
    @inference.CacheScope::from_ascii(b"tenant-a", inference),
  )
  let write = workspace.begin_text_with_sampling_scalars(
    byte_count=1,
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(1UL),
    max_new_tokens=1,
    context_ceiling=2,
    sampling_mode=Greedy,
    temperature=1.0,
    has_top_k=false,
    top_k=0,
    has_top_p=false,
    top_p=0.0,
    sampling_seed_value=1UL,
    stop_token_count=1,
    stop_string_count=1,
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(10UL, inference),
  )
  write.push_byte(b'x')
  try write.finish() catch {
    LunaFramedTextRequestNotReady => ()
    _ => fail("wrong incomplete-stop error")
  } noraise {
    _ => fail("incomplete stop payload was published")
  }
  write.push_stop_token(1)
  try
    write.push_stop_string_length(inference.max_stop_string_bytes() + 1)
  catch {
    LunaFramedTextRequestInvalidRange => ()
    _ => fail("wrong oversized-stop error")
  } noraise {
    _ => fail("oversized stop string was accepted")
  }
  write.abort()
}
