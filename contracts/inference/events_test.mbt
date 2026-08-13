///|
test "accepted event pins effective limits and model identity" {
  let cfg = limits()
  let effective = @inference.EffectiveLimits::new(
    max_new_tokens=4,
    context_ceiling=10,
    cfg,
  )
  let accepted = @inference.AcceptedEvent::new(
    model_identity=model_identity(),
    effective_limits=effective,
  )
  assert_eq(accepted.model_identity(), model_identity())
  assert_eq(accepted.effective_limits().max_new_tokens(), 4)
}

///|
test "token event validates UTF-8 token identity and position bounds" {
  let cfg = limits()
  let delta = @inference.DecodedDelta::from_utf8(b"hello", cfg)
  let token = @inference.TokenEvent::new(
    token_id=42,
    decoded_delta=Some(delta),
    position=0,
    cfg,
  )
  assert_eq(token.token_id(), 42)
  assert_eq(token.position(), 0)
  assert_eq(token.decoded_delta().unwrap().text(), "hello")
  try @inference.DecodedDelta::from_utf8([0xff], cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=DecodedDelta, issue=InvalidFormat, index=0)",
      )
  } noraise {
    _ => fail("expected decoded UTF-8 rejection")
  }
  try
    @inference.TokenEvent::new(token_id=1, decoded_delta=None, position=-1, cfg)
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=TokenPosition, issue=OutOfRange, index=0)",
      )
  } noraise {
    _ => fail("expected negative position rejection")
  }
  try
    @inference.TokenEvent::new(token_id=1, decoded_delta=None, position=6, cfg)
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=TokenPosition, issue=OutOfRange, index=0)",
      )
  } noraise {
    _ => fail("expected position upper-bound rejection")
  }
}

///|
test "usage construction and advancement enforce monotonic accounting" {
  let cfg = limits()
  let initial = @inference.Usage::initial(
    input_tokens=4,
    cached_input_tokens=2,
    cfg,
  )
  let advanced = initial.advance(output_tokens=3, cfg)
  assert_eq(advanced.input_tokens(), 4)
  assert_eq(advanced.cached_input_tokens(), 2)
  assert_eq(advanced.output_tokens(), 3)
  assert_eq(advanced.total_tokens(), 7)
  try advanced.advance(output_tokens=2, cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=UsageOutput, issue=NotMonotonic, index=0)",
      )
  } noraise {
    _ => fail("expected usage regression rejection")
  }
  try
    @inference.Usage::initial(input_tokens=3, cached_input_tokens=4, cfg)
  catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=UsageCachedInput, issue=Inconsistent, index=0)",
      )
  } noraise {
    _ => fail("expected cached usage rejection")
  }
  try initial.advance(output_tokens=9, cfg) catch {
    error =>
      debug_inspect(
        error,
        content="Invalid(field=UsageOutput, issue=OutOfRange, index=0)",
      )
  } noraise {
    _ => fail("expected output usage bound rejection")
  }
}

///|
test "terminal events expose bounded reasons usage and retryability" {
  let cfg = limits()
  let usage = @inference.Usage::initial(
    input_tokens=2,
    cached_input_tokens=0,
    cfg,
  ).advance(output_tokens=1, cfg)
  let completed = @inference.CompletedEvent::with_final_delta(
    finish_reason=StopSequence,
    final_usage=usage,
    final_delta=@inference.DecodedDelta::from_utf8(b"tail", cfg),
    cfg,
  )
  assert_eq(completed.finish_reason(), StopSequence)
  assert_eq(completed.final_usage().total_tokens(), 3)
  assert_eq(completed.final_delta().unwrap().text(), "tail")
  let failed = @inference.FailedEvent::new(
    code=@inference.PublicErrorCode::from_ascii(b"worker_unavailable"),
    retryable=true,
  )
  assert_eq(failed.code().as_string(), "worker_unavailable")
  assert_true(failed.retryable())
  for code in [b"bad/path", b"has space", Bytes::make(65, b'a')] {
    try @inference.PublicErrorCode::from_ascii(code) catch {
      error => assert_true(error is Invalid(..))
    } noraise {
      _ => fail("expected payload-unsafe public error code rejection")
    }
  }
}

///|
test "completed final delta is readmitted against the exact event envelope" {
  let broad = limits()
  let narrow = @inference.InferenceLimits::new(
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
    max_decoded_delta_bytes=4,
    max_deadline_millis=1000UL,
    max_top_k=100,
    max_temperature=4.0,
  )
  let delta = @inference.DecodedDelta::from_utf8(b"123456", broad)
  let usage = @inference.Usage::initial(
    input_tokens=1,
    cached_input_tokens=0,
    narrow,
  )
  try
    @inference.CompletedEvent::with_final_delta(
      finish_reason=TokenLimit,
      final_usage=usage,
      final_delta=delta,
      narrow,
    )
  catch {
    Invalid(field=DecodedDelta, issue=TooLong, index=0) => ()
    _ => fail("unexpected completed final delta error")
  } noraise {
    _ => fail("expected cross-envelope final delta rejection")
  }
}

///|
test "stream event vocabulary remains closed and typed" {
  let cfg = limits()
  let event : @inference.StreamEvent = Token(
    @inference.TokenEvent::new(token_id=3, decoded_delta=None, position=0, cfg),
  )
  match event {
    Token(token) => assert_eq(token.position(), 0)
    _ => fail("unexpected event")
  }
}
