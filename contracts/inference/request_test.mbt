///|
fn limits() -> @inference.InferenceLimits raise {
  limits_with_temperature(4.0)
}

///|
fn limits_with_temperature(
  max_temperature : Double,
) -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=16,
    max_input_tokens=8,
    max_new_tokens=6,
    max_context_tokens=12,
    max_token_id=99,
    max_stop_token_ids=3,
    max_stop_strings=2,
    max_stop_string_bytes=5,
    max_trace_bytes=8,
    max_cache_scope_bytes=8,
    max_decoded_delta_bytes=8,
    max_deadline_millis=1000UL,
    max_top_k=100,
    max_temperature~,
  )
}

///|
fn model_identity() -> @spec.ModelIdentity raise {
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
fn cache() -> @inference.CachePolicy raise {
  @inference.CachePolicy::new(
    @inference.CacheScope::from_ascii(b"scope-1", limits()),
    ReadWrite,
  )
}

///|
fn empty_stops() -> @inference.StopConditions raise {
  @inference.StopConditions::new([], [], limits())
}

///|
test "protocol and payload-safe request identities are explicit" {
  assert_eq(@inference.ProtocolVersion::v1().wire_code(), 1)
  assert_eq(@inference.ProtocolVersion::from_wire(1).wire_code(), 1)
  assert_eq(@inference.RequestId::new(42UL).value(), 42UL)
  assert_eq(@inference.RequestGeneration::initial().next().value(), 1UL)
  assert_eq(model_identity().content().as_hex().length(), 64)
  try @inference.ProtocolVersion::from_wire(2) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=ProtocolVersion, issue=Unsupported, index=0)",
      )
  } noraise {
    _ => fail("expected unsupported protocol rejection")
  }
  try @inference.RequestId::new(0UL) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=RequestId, issue=OutOfRange, index=0)",
      )
  } noraise {
    _ => fail("expected request identity rejection")
  }
  try
    @inference.RequestGeneration::from_wire(0xffffffffffffffffUL).next()
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=RequestGeneration, issue=OutOfRange, index=0)",
      )
  } noraise {
    _ => fail("expected generation wrap rejection")
  }
}

///|
test "token input is copied before caller mutation" {
  let source = [1, 2, 3]
  let input = @inference.TokenBuffer::new(source, limits())
  source[0] = 77
  assert_eq(input.token_at(0), 1)
  assert_eq(input.length(), 3)
}

///|
test "input token and text bounds fail before ownership" {
  let cfg = limits()
  try @inference.Input::from_token_ids([], cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=InputToken, issue=Empty, index=0)",
      )
  } noraise {
    _ => fail("expected empty token rejection")
  }
  try @inference.Input::from_token_ids(Array::make(9, 1), cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=InputToken, issue=TooMany, index=8)",
      )
  } noraise {
    _ => fail("expected token count rejection")
  }
  try @inference.Input::from_token_ids([1, -1], cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=InputToken, issue=OutOfRange, index=1)",
      )
  } noraise {
    _ => fail("expected negative token rejection")
  }
  try @inference.Input::from_token_ids([100], cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=InputToken, issue=OutOfRange, index=0)",
      )
  } noraise {
    _ => fail("expected vocabulary bound rejection")
  }
  try @inference.Input::from_utf8(b"", cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=InputText, issue=Empty, index=0)",
      )
  } noraise {
    _ => fail("expected empty text rejection")
  }
  try @inference.Input::from_utf8(Bytes::make(17, b'a'), cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=InputText, issue=TooLong, index=0)",
      )
  } noraise {
    _ => fail("expected text byte bound rejection")
  }
  try @inference.Input::from_utf8([0xff], cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=InputText, issue=InvalidFormat, index=0)",
      )
  } noraise {
    _ => fail("expected malformed UTF-8 rejection")
  }
}

///|
test "sampling rejects non-finite and invalid filters" {
  let cfg = limits()
  for
    temperature in [
      @double.not_a_number, @double.infinity, @double.neg_infinity,
    ] {
    try @inference.SamplingParameters::sample(temperature~, cfg) catch {
      error =>
        debug_inspect(
          error,
          content="Invalid(field=Temperature, issue=NonFinite, index=0)",
        )
    } noraise {
      _ => fail("expected non-finite temperature rejection")
    }
  }
  try
    @inference.SamplingParameters::sample(temperature=1.0, top_k=0, cfg)
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=TopK, issue=OutOfRange, index=0)",
      )
  } noraise {
    _ => fail("expected top-k rejection")
  }
  try @inference.SamplingParameters::sample(temperature=4.1, cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=Temperature, issue=OutOfRange, index=0)",
      )
  } noraise {
    _ => fail("expected temperature magnitude rejection")
  }
  for top_p in [@double.not_a_number, @double.infinity] {
    try
      @inference.SamplingParameters::sample(temperature=1.0, top_p~, cfg)
    catch {
      error =>
        debug_inspect(
          error,
          content="Invalid(field=TopP, issue=NonFinite, index=0)",
        )
    } noraise {
      _ => fail("expected non-finite top-p rejection")
    }
  }
  try
    @inference.SamplingParameters::sample(temperature=1.0, top_p=1.1, cfg)
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=TopP, issue=OutOfRange, index=0)",
      )
  } noraise {
    _ => fail("expected top-p range rejection")
  }
  let sampled = @inference.SamplingParameters::sample(
    temperature=0.75,
    top_k=10,
    top_p=0.9,
    cfg,
  )
  assert_eq(sampled.mode(), Sample)
  assert_eq(sampled.top_k(), Some(10))
  assert_eq(sampled.top_p(), Some(0.9))
}

///|
test "configured sampling magnitude is finite positive and hard bounded" {
  assert_eq(limits_with_temperature(100.0).max_temperature(), 100.0)
  for maximum in [0.0, 100.1, @double.not_a_number, @double.infinity] {
    try limits_with_temperature(maximum) catch {
      error =>
        debug_inspect(
          error,
          content="Invalid(field=Limit, issue=OutOfRange, index=0)",
        )
    } noraise {
      _ => fail("expected maximum temperature limit rejection")
    }
  }
}

///|
test "stop conditions reject duplicates IDs invalid UTF-8 and every bound" {
  let cfg = limits()
  try @inference.StopConditions::new([1, 1], [], cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=StopToken, issue=Duplicate, index=1)",
      )
  } noraise {
    _ => fail("expected duplicate stop token rejection")
  }
  try @inference.StopConditions::new([100], [], cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=StopToken, issue=OutOfRange, index=0)",
      )
  } noraise {
    _ => fail("expected invalid stop token rejection")
  }
  try @inference.StopConditions::new([1, 2, 3, 4], [], cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=StopToken, issue=TooMany, index=3)",
      )
  } noraise {
    _ => fail("expected stop token count rejection")
  }
  try @inference.StopConditions::new([], [b"x", b"x"], cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=StopString, issue=Duplicate, index=1)",
      )
  } noraise {
    _ => fail("expected duplicate stop string rejection")
  }
  try @inference.StopConditions::new([], [b""], cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=StopString, issue=Empty, index=0)",
      )
  } noraise {
    _ => fail("expected empty stop string rejection")
  }
  try @inference.StopConditions::new([], [[0xff]], cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=StopString, issue=InvalidFormat, index=0)",
      )
  } noraise {
    _ => fail("expected malformed stop string rejection")
  }
  try @inference.StopConditions::new([], [b"123456"], cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=StopString, issue=TooLong, index=0)",
      )
  } noraise {
    _ => fail("expected stop string byte bound rejection")
  }
  try @inference.StopConditions::new([], [b"a", b"b", b"c"], cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=StopString, issue=TooMany, index=2)",
      )
  } noraise {
    _ => fail("expected stop string count rejection")
  }
}

///|
test "cache trace and relative deadline use bounded payload-safe identities" {
  let cfg = limits()
  assert_eq(
    @inference.DeadlineBudget::from_milliseconds(500UL, cfg).milliseconds(),
    500UL,
  )
  for millis in [0UL, 1001UL] {
    try @inference.DeadlineBudget::from_milliseconds(millis, cfg) catch {
      error =>
        debug_inspect(
          error,
          content="Invalid(field=DeadlineBudget, issue=OutOfRange, index=0)",
        )
    } noraise {
      _ => fail("expected deadline rejection")
    }
  }
  try @inference.CacheScope::from_ascii(b"bad/sco", cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=CacheScope, issue=InvalidFormat, index=0)",
      )
  } noraise {
    _ => fail("expected cache scope grammar rejection")
  }
  try @inference.TraceCorrelation::from_ascii(b"123456789", cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=TraceCorrelation, issue=TooLong, index=0)",
      )
  } noraise {
    _ => fail("expected trace bound rejection")
  }
}

///|
test "canonical request validates exact token context without overflow" {
  let cfg = limits()
  let request = @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(7UL),
    model_identity=model_identity(),
    input=@inference.Input::from_token_ids([1, 2, 3], cfg),
    max_new_tokens=4,
    context_ceiling=7,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(17UL),
    stops=empty_stops(),
    stream_preference=Streaming,
    deadline=@inference.DeadlineBudget::from_milliseconds(250UL, cfg),
    cache=cache(),
    trace=Some(@inference.TraceCorrelation::from_ascii(b"trace-1", cfg)),
    limits=cfg,
  )
  assert_eq(request.request_id().value(), 7UL)
  assert_eq(request.max_new_tokens(), 4)
  assert_eq(request.context_ceiling(), 7)
  assert_eq(request.sampling_seed().value(), 17UL)
  try
    @inference.GenerateRequest::new(
      protocol_version=@inference.ProtocolVersion::v1(),
      request_id=@inference.RequestId::new(8UL),
      model_identity=model_identity(),
      input=@inference.Input::from_token_ids([1, 2, 3, 4], cfg),
      max_new_tokens=4,
      context_ceiling=7,
      sampling=@inference.SamplingParameters::greedy(),
      sampling_seed=@inference.SamplingSeed::new(18UL),
      stops=empty_stops(),
      stream_preference=Buffered,
      deadline=@inference.DeadlineBudget::from_milliseconds(250UL, cfg),
      cache=cache(),
      trace=None,
      limits=cfg,
    )
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=ContextCeiling, issue=Inconsistent, index=0)",
      )
  } noraise {
    _ => fail("expected exact token context rejection")
  }
}

///|
test "token-id streaming rejects text stop semantics" {
  let cfg = limits()
  try
    @inference.GenerateRequest::new(
      protocol_version=@inference.ProtocolVersion::v1(),
      request_id=@inference.RequestId::new(9UL),
      model_identity=model_identity(),
      input=@inference.Input::from_token_ids([1, 2, 3], cfg),
      max_new_tokens=4,
      context_ceiling=7,
      sampling=@inference.SamplingParameters::greedy(),
      sampling_seed=@inference.SamplingSeed::new(19UL),
      stops=@inference.StopConditions::new([], [b"END"], cfg),
      stream_preference=TokenIdsStreaming,
      deadline=@inference.DeadlineBudget::from_milliseconds(250UL, cfg),
      cache=cache(),
      trace=None,
      limits=cfg,
    )
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=StopString, issue=Inconsistent, index=0)",
      )
  } noraise {
    _ => fail("expected token-id streaming stop-string rejection")
  }
}

///|
test "admission deadline is captured once at receipt with checked arithmetic" {
  let cfg = limits()
  let budget = @inference.DeadlineBudget::from_milliseconds(250UL, cfg)
  let deadline = @inference.AdmissionDeadline::from_receipt(1000UL, budget)
  assert_eq(deadline.at_millis(), 1250UL)
  assert_false(deadline.is_expired(1249UL))
  assert_true(deadline.is_expired(1250UL))
  try
    @inference.AdmissionDeadline::from_receipt(0xffffffffffffff80UL, budget)
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=DeadlineBudget, issue=OutOfRange, index=0)",
      )
  } noraise {
    _ => fail("overflowing receipt deadline was admitted")
  }
}

///|
test "sampling seed is canonical and rejects the absorbing state" {
  assert_eq(@inference.SamplingSeed::new(42UL).value(), 42UL)
  try @inference.SamplingSeed::new(0UL) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=SamplingSeed, issue=OutOfRange, index=0)",
      )
  } noraise {
    _ => fail("zero sampling seed was admitted")
  }
}
