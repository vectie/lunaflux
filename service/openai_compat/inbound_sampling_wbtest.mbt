///|
fn inbound_sampling_body() -> String {
  "{\"model\":\"model-x\",\"temperature\":7.5e-1,\"top_k\":12,\"top_p\":9e-1,\"seed\":99,\"stop\":[\"END\",\"stop\"],\"stop_token_ids\":[2,7],\"input\":\"hello moon\",\"stream\":true}"
}

///|
fn inbound_sampling_expected() -> @inference.GenerateRequest raise {
  let inference = openai_wb_fixture_limits()
  @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(42UL),
    model_identity=inbound_identity(),
    input=@inference.Input::from_utf8(b"hello moon", inference),
    max_new_tokens=4,
    context_ceiling=16,
    sampling=@inference.SamplingParameters::sample(
      temperature=0.75,
      top_k=12,
      top_p=0.9,
      inference,
    ),
    sampling_seed=@inference.SamplingSeed::new(99UL),
    stops=@inference.StopConditions::new([2, 7], [b"END", b"stop"], inference),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(5000UL, inference),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"openai", inference),
      Disabled,
    ),
    trace=None,
    limits=inference,
  )
}

///|
fn inbound_sampling_workspace(
  inference : @inference.InferenceLimits,
  sampling : @inference.SamplingParameters,
  seed : @inference.SamplingSeed,
) -> LunaOpenAIInboundWorkspace raise {
  let model = inbound_bytes("model-x")
  LunaOpenAIInboundWorkspace::new(
    inference,
    LunaOpenAICompatStepBudget::new(17),
    LunaOpenAIInboundLimits::new(max_messages=8),
    inbound_template(inference),
    model,
    model_alias_offset=0,
    model_alias_length=model.length(),
    inbound_identity(),
    @inference.CacheScope::from_ascii(b"openai", inference),
    max_new_tokens=4,
    context_ceiling=16,
    sampling,
    seed,
    @inference.DeadlineBudget::from_milliseconds(5000UL, inference),
  )
}

///|
test "Luna OpenAI sampling and stops are canonical across budgets" {
  let inference = openai_wb_fixture_limits()
  let expected_frame = @framed_wire.RequestFrameBuffer::new(
    @framed_wire.FramedWireLimits::new(inference, max_frame_bytes=4096),
  ).encode(inbound_sampling_expected())
  let expected : FixedArray[Byte] = FixedArray::make(expected_frame.length(), 0)
  expected_frame.copy_to(expected, destination_offset=0)
  let mut expected_total = 0UL
  for budget in [1, 17, 65536] {
    let (actual, total) = inbound_encode(
      inbound_workspace(inference, budget),
      inbound_http_view(1, inbound_sampling_body()),
    )
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
test "Luna OpenAI absent sampling fields preserve validated startup defaults" {
  let inference = openai_wb_fixture_limits()
  let defaults = @inference.SamplingParameters::sample(
    temperature=0.625,
    top_k=9,
    top_p=0.875,
    inference,
  )
  let (frame, _) = inbound_encode(
    inbound_sampling_workspace(
      inference,
      defaults,
      @inference.SamplingSeed::new(123UL),
    ),
    inbound_http_view(
      2, "{\"model\":\"model-x\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}",
    ),
  )
  let request = @framed_wire.RequestFrameBuffer::new(
      @framed_wire.FramedWireLimits::new(inference, max_frame_bytes=4096),
    )
    .load(frame, frame.length())
    .request()
  assert_eq(request.sampling().mode(), Sample)
  assert_eq(request.sampling().temperature(), 0.625)
  assert_eq(request.sampling().top_k(), Some(9))
  assert_eq(request.sampling().top_p(), Some(0.875))
  assert_eq(request.sampling_seed().value(), 123UL)
  assert_eq(request.stops().token_ids().length(), 0)
  assert_eq(request.stops().strings().length(), 0)
}

///|
test "Luna OpenAI optional sampling fields fail before frame publication" {
  let bodies : FixedArray[String] = [
    "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"seed\":1,\"seed\":2}",
    "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"temperature\":1e}",
    "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"temperature\":1e401}",
    "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"temperature\":9}",
    "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"top_k\":0}", "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"top_k\":1.0}",
    "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"top_p\":0}", "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"top_p\":1.0001}",
    "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"seed\":-1}", "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"seed\":1.0}",
    "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"seed\":18446744073709551616}",
    "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"stop\":[\"\"]}", "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"stop\":[\"x\",\"x\"]}",
    "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"stop_token_ids\":[2,2]}",
    "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"stop_token_ids\":[256]}",
  ]
  let inference = openai_wb_fixture_limits()
  for body in bodies {
    let workspace = inbound_workspace(inference, 1)
    let work = workspace.begin(
      inbound_http_view(1, body),
      @inference.RequestId::new(55UL),
    )
    while work.state() == LunaOpenAIInboundPending {
      ignore(work.progress())
    }
    assert_eq(work.state(), LunaOpenAIInboundFailed)
    assert_eq(work.failure(), LunaOpenAICompatInvalid(LunaOpenAICompatJson))
    assert_eq(workspace.handoff_writes.length(), 0)
    assert_eq(workspace.handoff_works.length(), 0)
    assert_eq(workspace.handoff_leases.length(), 0)
    work.abort()
  }
}

///|
test "Luna OpenAI sampling scalar inclusive maxima are preserved exactly" {
  let inference = openai_wb_fixture_limits()
  let (frame, _) = inbound_encode(
    inbound_workspace(inference, 17),
    inbound_http_view(
      1, "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"temperature\":8,\"top_k\":256,\"top_p\":1,\"seed\":18446744073709551615}",
    ),
  )
  let request = @framed_wire.RequestFrameBuffer::new(
      @framed_wire.FramedWireLimits::new(inference, max_frame_bytes=4096),
    )
    .load(frame, frame.length())
    .request()
  assert_eq(request.sampling().mode(), Sample)
  assert_eq(request.sampling().temperature(), 8.0)
  assert_eq(request.sampling().top_k(), Some(256))
  assert_eq(request.sampling().top_p(), Some(1.0))
  assert_eq(request.sampling_seed().value(), 18446744073709551615UL)
}

///|
test "Luna OpenAI stop string and token capacity boundaries are exact" {
  let (single_frame, _) = inbound_encode(
    inbound_workspace(openai_wb_fixture_limits(), 17),
    inbound_http_view(
      1, "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"stop\":\"END\"}",
    ),
  )
  let single = @framed_wire.RequestFrameBuffer::new(
      @framed_wire.FramedWireLimits::new(
        openai_wb_fixture_limits(),
        max_frame_bytes=4096,
      ),
    )
    .load(single_frame, single_frame.length())
    .request()
  assert_eq(single.stops().strings().length(), 1)
  assert_eq(single.stops().strings()[0], "END")
  let exact = "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"stop\":[\"1234567890123456\",\"a\",\"b\"],\"stop_token_ids\":[0,1,2,255]}"
  let (frame, _) = inbound_encode(
    inbound_workspace(openai_wb_fixture_limits(), 17),
    inbound_http_view(1, exact),
  )
  let request = @framed_wire.RequestFrameBuffer::new(
      @framed_wire.FramedWireLimits::new(
        openai_wb_fixture_limits(),
        max_frame_bytes=4096,
      ),
    )
    .load(frame, frame.length())
    .request()
  assert_eq(request.stops().token_ids().length(), 4)
  assert_eq(request.stops().strings().length(), 3)
  inbound_expect_json_failure(
    1, "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"stop\":[\"12345678901234567\"]}",
  )
  inbound_expect_json_failure(
    1, "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"stop_token_ids\":[0,1,2,3,4]}",
  )
}

///|
test "Luna OpenAI frame scalars replay the deterministic device sampler" {
  let inference = openai_wb_fixture_limits()
  let (frame, _) = inbound_encode(
    inbound_workspace(inference, 17),
    inbound_http_view(1, inbound_sampling_body()),
  )
  let request = @framed_wire.RequestFrameBuffer::new(
      @framed_wire.FramedWireLimits::new(inference, max_frame_bytes=4096),
    )
    .load(frame, frame.length())
    .request()
  let logits : FixedArray[Double] = [0.1, 1.7, -0.5, 1.2, 0.9]
  let limits = @sampling.SamplingLimits::new(1, logits.length())
  for sample_index in 0..<8 {
    let object_selection = @sampling.stochastic_sample_at(
      logits[:],
      request.sampling(),
      limits,
      request.sampling_seed(),
      sample_index,
      @sampling.SamplerScratch::new(logits.length(), limits),
    )
    let scalar_selection = @sampling.stochastic_sample_at_scalars(
      logits[:],
      temperature=request.sampling().temperature(),
      has_top_k=request.sampling().top_k() is Some(_),
      top_k=request.sampling().top_k().unwrap_or(0),
      has_top_p=request.sampling().top_p() is Some(_),
      top_p=request.sampling().top_p().unwrap_or(0.0),
      limits,
      request.sampling_seed(),
      sample_index,
      @sampling.SamplerScratch::new(logits.length(), limits),
    )
    assert_eq(object_selection.token_id(), scalar_selection.token_id())
    assert_eq(object_selection.logit(), scalar_selection.logit())
  }
}
