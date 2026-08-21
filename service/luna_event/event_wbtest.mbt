///|
fn luna_fixture_limits(
  max_delta? : Int = 16,
) -> @inference.InferenceLimits raise {
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
    max_decoded_delta_bytes=max_delta,
    max_deadline_millis=10000UL,
    max_top_k=256,
    max_temperature=8.0,
  )
}

///|
fn luna_fixture_identity() -> @spec.ModelIdentity raise {
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
fn luna_assert_bytes(actual : FixedArray[Byte], expected : Bytes) -> Unit raise {
  assert_eq(actual.length(), expected.length())
  for index in 0..<actual.length() {
    assert_eq(actual[index], expected[index])
  }
}

///|
fn luna_usage_view(owner : LunaEventOwner) -> LunaUsageEventView raise {
  owner.view().usage()
}

///|
test "accepted view is exact and becomes stale after retirement" {
  let limits = luna_fixture_limits()
  let owner = LunaEventOwner::new(limits)
  try owner.view() catch {
    LunaEventInvalid(LunaEventLifecycle) => ()
    _ => fail("unexpected empty owner error")
  } noraise {
    _ => fail("empty owner published a view")
  }
  let effective = @inference.EffectiveLimits::new(
    max_new_tokens=4,
    context_ceiling=16,
    limits,
  )
  owner.publish_accepted(luna_fixture_identity(), effective)
  let event = owner.view()
  assert_eq(event.kind(), LunaAcceptedEventKind)
  let accepted = event.accepted()
  assert_eq(accepted.model_identity(), luna_fixture_identity())
  assert_eq(accepted.effective_limits(), effective)
  owner.retire(event)
  assert_false(owner.has_event())
  try accepted.effective_limits() catch {
    LunaEventInvalid(LunaEventEpoch) => ()
    _ => fail("unexpected stale Accepted error")
  } noraise {
    _ => fail("retired Accepted view remained live")
  }
}

///|
test "foreign view cannot retire another owner" {
  let limits = luna_fixture_limits()
  let first = LunaEventOwner::new(limits)
  let second = LunaEventOwner::new(limits)
  first.publish_usage(input_tokens=3, cached_input_tokens=1, output_tokens=2)
  second.publish_usage(input_tokens=4, cached_input_tokens=0, output_tokens=1)
  let foreign = first.view()
  try second.retire(foreign) catch {
    LunaEventInvalid(LunaEventEpoch) => ()
    _ => fail("unexpected foreign retirement error")
  } noraise {
    _ => fail("foreign view retired another owner")
  }
  assert_eq(luna_usage_view(first).total_tokens(), 5)
  assert_eq(luna_usage_view(second).total_tokens(), 5)
  let empty = FixedArray::make(0, b'\x00')
  try
    second.replace_usage_with_completed(
      luna_usage_view(first),
      TokenLimit,
      empty,
      tail_offset=0,
      tail_length=0,
    )
  catch {
    LunaEventInvalid(LunaEventEpoch) => ()
    _ => fail("unexpected foreign Usage replacement error")
  } noraise {
    _ => fail("foreign Usage replaced another owner's event")
  }
  try second.view().token() catch {
    LunaEventInvalid(LunaEventKind) => ()
    _ => fail("unexpected wrong-kind projection error")
  } noraise {
    _ => fail("Usage projected as Token")
  }
}

///|
test "token status distinguishes payload, credit, and epoch rejection" {
  let limits = luna_fixture_limits(max_delta=4)
  let owner = LunaEventOwner::new(limits)
  let invalid_utf8 : FixedArray[Byte] = [b'\xff']
  assert_eq(
    owner.publish_token_status(
      token_id=1,
      position=0,
      invalid_utf8,
      source_offset=0,
      length=1,
    ),
    LunaEventPublishPayloadInvalid,
  )
  assert_false(owner.has_event())
  let source : FixedArray[Byte] = [b'a', b'\xe2', b'\x82', b'\xac']
  assert_eq(
    owner.publish_token_status(
      token_id=7,
      position=2,
      source,
      source_offset=0,
      length=4,
    ),
    LunaEventPublishReady,
  )
  assert_eq(
    owner.publish_token_status(
      token_id=8,
      position=3,
      source,
      source_offset=0,
      length=1,
    ),
    LunaEventPublishCreditUnavailable,
  )
  let event = owner.view()
  assert_eq(event.kind(), LunaTokenEventKind)
  let token = event.token()
  assert_eq(token.token_id(), 7)
  assert_eq(token.position(), 2)
  assert_eq(token.delta_length(), 4)
  let copied = FixedArray::make(4, b'\x00')
  assert_eq(token.copy_delta_to(copied, destination_offset=0), 4)
  luna_assert_bytes(copied, b"a\xe2\x82\xac")
  let small = FixedArray::make(3, b'x')
  try token.copy_delta_to(small, destination_offset=0) catch {
    LunaEventInvalid(LunaEventDestination) => ()
    _ => fail("unexpected small destination error")
  } noraise {
    _ => fail("small token destination passed")
  }
  owner.retire(event)
  owner.epoch = 0xffffffffffffffffUL
  assert_eq(
    owner.publish_token_status(
      token_id=1,
      position=0,
      source,
      source_offset=0,
      length=1,
    ),
    LunaEventPublishEpochExhausted,
  )
  assert_eq(owner.epoch, 0xffffffffffffffffUL)
  assert_false(owner.has_event())
}

///|
test "epoch-capacity preflight is exact near UInt64 exhaustion and read-only" {
  let owner = LunaEventOwner::new(luna_fixture_limits())
  owner.epoch = 0xfffffffffffffffcUL
  assert_true(owner.has_epoch_capacity(3))
  assert_false(owner.has_epoch_capacity(4))
  assert_false(owner.has_epoch_capacity(-1))
  assert_eq(owner.epoch, 0xfffffffffffffffcUL)
  assert_false(owner.has_event())
  owner.epoch = 0xffffffffffffffffUL
  assert_true(owner.has_epoch_capacity(0))
  assert_false(owner.has_epoch_capacity(1))
  assert_eq(owner.epoch, 0xffffffffffffffffUL)
  assert_false(owner.has_event())
}

///|
test "Usage to Completed replacement is atomic and advances epoch" {
  let limits = luna_fixture_limits()
  let owner = LunaEventOwner::new(limits)
  owner.publish_usage(input_tokens=4, cached_input_tokens=1, output_tokens=2)
  let usage = luna_usage_view(owner)
  assert_eq(usage.input_tokens(), 4)
  assert_eq(usage.cached_input_tokens(), 1)
  assert_eq(usage.output_tokens(), 2)
  assert_eq(usage.total_tokens(), 6)
  let malformed : FixedArray[Byte] = [b'\xc0', b'\x80']
  try
    owner.replace_usage_with_completed(
      usage,
      TokenLimit,
      malformed,
      tail_offset=0,
      tail_length=2,
    )
  catch {
    LunaEventInvalid(LunaEventPayload) => ()
    _ => fail("unexpected invalid Completed replacement error")
  } noraise {
    _ => fail("invalid Completed tail replaced Usage")
  }
  assert_eq(usage.total_tokens(), 6)
  let tail : FixedArray[Byte] = [b'o', b'k']
  let completed = owner.replace_usage_with_completed(
    usage,
    TokenLimit,
    tail,
    tail_offset=0,
    tail_length=2,
  )
  try usage.total_tokens() catch {
    LunaEventInvalid(LunaEventEpoch) => ()
    _ => fail("unexpected replaced Usage error")
  } noraise {
    _ => fail("replaced Usage view remained live")
  }
  assert_eq(completed.finish_reason(), TokenLimit)
  assert_eq(completed.total_tokens(), 6)
  assert_eq(completed.final_delta_length(), 2)
  let copied = FixedArray::make(2, b'\x00')
  assert_eq(completed.copy_final_delta_to(copied, destination_offset=0), 2)
  luna_assert_bytes(copied, b"ok")
}

///|
test "Usage to Failed replacement preserves Usage on invalid public code" {
  let owner = LunaEventOwner::new(luna_fixture_limits(max_delta=1))
  owner.publish_usage(input_tokens=2, cached_input_tokens=0, output_tokens=1)
  let usage = luna_usage_view(owner)
  let invalid : FixedArray[Byte] = [b'b', b'a', b'd', b' ']
  try
    owner.replace_usage_with_failed(
      usage,
      invalid,
      code_offset=0,
      code_length=4,
      retryable=false,
    )
  catch {
    LunaEventInvalid(LunaEventPayload) => ()
    _ => fail("unexpected invalid Failed replacement error")
  } noraise {
    _ => fail("invalid public code replaced Usage")
  }
  assert_eq(usage.output_tokens(), 1)
  let code = FixedArray::make(64, b'a')
  let failed = owner.replace_usage_with_failed(
    usage,
    code,
    code_offset=0,
    code_length=64,
    retryable=true,
  )
  assert_true(failed.retryable())
  assert_eq(failed.code_length(), 64)
  let copied = FixedArray::make(64, b'\x00')
  assert_eq(failed.copy_code_to(copied, destination_offset=0), 64)
  for byte in copied {
    assert_eq(byte, b'a')
  }
}

///|
test "invalid scalar and range publications preserve free credit" {
  let limits = luna_fixture_limits(max_delta=4)
  let owner = LunaEventOwner::new(limits)
  try
    owner.publish_usage(input_tokens=32, cached_input_tokens=0, output_tokens=1)
  catch {
    LunaEventInvalid(LunaEventScalar) => ()
    _ => fail("unexpected invalid Usage error")
  } noraise {
    _ => fail("inconsistent Usage passed")
  }
  let tail : FixedArray[Byte] = [b'a', b'b', b'c', b'd', b'e']
  try
    owner.publish_completed(
      StopSequence,
      input_tokens=2,
      cached_input_tokens=0,
      output_tokens=1,
      tail,
      tail_offset=0,
      tail_length=5,
    )
  catch {
    LunaEventInvalid(LunaEventLength) => ()
    _ => fail("unexpected oversized Completed error")
  } noraise {
    _ => fail("oversized Completed tail passed")
  }
  let invalid_code : FixedArray[Byte] = [b'/', b'x']
  try
    owner.publish_failed(
      invalid_code,
      code_offset=0,
      code_length=2,
      retryable=false,
    )
  catch {
    LunaEventInvalid(LunaEventPayload) => ()
    _ => fail("unexpected invalid code error")
  } noraise {
    _ => fail("invalid public code passed")
  }
  assert_false(owner.has_event())
}

///|
test "discard invalidates a borrowed view without acknowledgement" {
  let owner = LunaEventOwner::new(luna_fixture_limits())
  let code : FixedArray[Byte] = [b'f', b'a', b'i', b'l']
  owner.publish_failed(code, code_offset=0, code_length=4, retryable=false)
  let failed = owner.view().failed()
  owner.discard()
  try failed.code_length() catch {
    LunaEventInvalid(LunaEventEpoch) => ()
    _ => fail("unexpected discarded view error")
  } noraise {
    _ => fail("discarded view remained live")
  }
}

///|
test "direct Completed and Failed publications expose typed bounded views" {
  let owner = LunaEventOwner::new(luna_fixture_limits())
  let tail : FixedArray[Byte] = [b'e', b'n', b'd']
  owner.publish_completed(
    StopSequence,
    input_tokens=3,
    cached_input_tokens=1,
    output_tokens=2,
    tail,
    tail_offset=0,
    tail_length=3,
  )
  let completed_event = owner.view()
  assert_eq(completed_event.kind(), LunaCompletedEventKind)
  let completed = completed_event.completed()
  assert_eq(completed.finish_reason(), StopSequence)
  assert_eq(completed.input_tokens(), 3)
  assert_eq(completed.cached_input_tokens(), 1)
  assert_eq(completed.output_tokens(), 2)
  assert_eq(completed.total_tokens(), 5)
  assert_eq(completed.final_delta_byte_at(0), b'e')
  owner.retire(completed_event)
  let code : FixedArray[Byte] = [b'w', b'o', b'r', b'k', b'e', b'r']
  owner.publish_failed(code, code_offset=0, code_length=6, retryable=true)
  let failed_event = owner.view()
  assert_eq(failed_event.kind(), LunaFailedEventKind)
  let failed = failed_event.failed()
  assert_true(failed.retryable())
  assert_eq(failed.code_length(), 6)
  assert_eq(failed.code_byte_at(5), b'r')
  let copied = FixedArray::make(6, b'\x00')
  assert_eq(failed.copy_code_to(copied, destination_offset=0), 6)
  luna_assert_bytes(copied, b"worker")
}

///|
test "payload byte access authenticates epoch before bounds" {
  let owner = LunaEventOwner::new(luna_fixture_limits())
  let delta : FixedArray[Byte] = [b'a', b'b']
  assert_eq(
    owner.publish_token_status(
      token_id=3,
      position=0,
      delta,
      source_offset=0,
      length=2,
    ),
    LunaEventPublishReady,
  )
  let token_event = owner.view()
  let token = token_event.token()
  assert_eq(token.delta_byte_at(0), b'a')
  assert_eq(token.delta_byte_at(1), b'b')
  try token.delta_byte_at(2) catch {
    LunaEventInvalid(LunaEventPayload) => ()
    _ => fail("unexpected live token byte bound error")
  } noraise {
    _ => fail("token byte beyond payload passed")
  }
  owner.retire(token_event)
  try token.delta_byte_at(-1) catch {
    LunaEventInvalid(LunaEventEpoch) => ()
    _ => fail("stale token byte did not authenticate before bounds")
  } noraise {
    _ => fail("stale token byte remained readable")
  }
  let effective = @inference.EffectiveLimits::new(
    max_new_tokens=4,
    context_ceiling=16,
    luna_fixture_limits(),
  )
  owner.publish_accepted(luna_fixture_identity(), effective)
  let accepted_event = owner.view()
  let accepted = accepted_event.accepted()
  assert_eq(accepted.content_digest_byte_at(0), b'0')
  assert_eq(accepted.plan_digest_byte_at(0), b'a')
  try accepted.plan_digest_byte_at(64) catch {
    LunaEventInvalid(LunaEventPayload) => ()
    _ => fail("unexpected accepted digest bound error")
  } noraise {
    _ => fail("accepted digest byte beyond payload passed")
  }
  owner.retire(accepted_event)
}
