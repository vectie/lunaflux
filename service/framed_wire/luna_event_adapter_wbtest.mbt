///|
fn luna_adapter_bytes(
  adapter : LunaFramedEventAdapter,
) -> FixedArray[Byte] raise {
  let storage = FixedArray::make(adapter.length(), b'\x00')
  ignore(adapter.copy_to(storage, destination_offset=0))
  storage
}

///|
fn luna_assert_adapter_matches(
  adapter : LunaFramedEventAdapter,
  expected : @inference.StreamEvent,
) -> Unit raise {
  assert_fixed_bytes_equal(
    luna_adapter_bytes(adapter),
    event_bytes(EventFrameBuffer::new(fixture_limits()).encode(expected)),
  )
}

///|
test "Luna adapter Accepted Usage and Failed bytes equal canonical event v2" {
  let inference = fixture_inference_limits()
  let adapter = LunaFramedEventAdapter::new(fixture_limits())
  let owner = @luna_event.LunaEventOwner::new(inference)
  let effective = @inference.EffectiveLimits::new(
    max_new_tokens=4,
    context_ceiling=16,
    inference,
  )
  owner.publish_accepted(fixture_identity(), effective)
  let accepted_view = owner.view()
  adapter.stage(accepted_view)
  let accepted_bytes = luna_adapter_bytes(adapter)
  assert_eq(
    @crypto.bytes_to_hex_string(
      @crypto.sha256(
        Bytes::makei(accepted_bytes.length(), index => accepted_bytes[index]),
      ),
    ),
    "341ae02aa5d3160f35595d8e4a091e1ef299679d08f1ce6c565bb35d8b288c71",
  )
  luna_assert_adapter_matches(
    adapter,
    Accepted(
      @inference.AcceptedEvent::new(
        model_identity=fixture_identity(),
        effective_limits=effective,
      ),
    ),
  )
  adapter.release()
  owner.retire(accepted_view)
  owner.publish_usage(input_tokens=3, cached_input_tokens=1, output_tokens=2)
  let usage_view = owner.view()
  adapter.stage(usage_view)
  let usage = @inference.Usage::initial(
    input_tokens=3,
    cached_input_tokens=1,
    inference,
  ).advance(output_tokens=2, inference)
  luna_assert_adapter_matches(adapter, Usage(usage))
  adapter.release()
  owner.retire(usage_view)
  let code : FixedArray[Byte] = [
    b'w', b'o', b'r', b'k', b'e', b'r', b'_', b'f', b'a', b'i', b'l', b'e', b'd',
  ]
  owner.publish_failed(
    code,
    code_offset=0,
    code_length=code.length(),
    retryable=true,
  )
  let failed_view = owner.view()
  adapter.stage(failed_view)
  luna_assert_adapter_matches(
    adapter,
    Failed(
      @inference.FailedEvent::new(
        code=@inference.PublicErrorCode::from_ascii(b"worker_failed"),
        retryable=true,
      ),
    ),
  )
  adapter.release()
  assert_eq(failed_view.failed().code_length(), code.length())
  owner.retire(failed_view)
  let terminal_code : FixedArray[Byte] = [
    b'o', b'u', b't', b'p', b'u', b't', b'_', b'i', b'n', b'v', b'a', b'l', b'i',
    b'd',
  ]
  owner.publish_failed(
    terminal_code,
    code_offset=0,
    code_length=terminal_code.length(),
    retryable=false,
  )
  let terminal_view = owner.view()
  adapter.stage(terminal_view)
  luna_assert_adapter_matches(
    adapter,
    Failed(
      @inference.FailedEvent::new(
        code=@inference.PublicErrorCode::from_ascii(b"output_invalid"),
        retryable=false,
      ),
    ),
  )
  adapter.release()
  owner.retire(terminal_view)
}

///|
test "Luna adapter Token and Completed payloads equal canonical event v2" {
  let inference = fixture_inference_limits()
  let adapter = LunaFramedEventAdapter::new(fixture_limits())
  let owner = @luna_event.LunaEventOwner::new(inference)
  let delta : FixedArray[Byte] = [b'x', b'o', b'k', 0xe4, 0xb8, 0xad]
  assert_eq(
    owner.publish_token_status(
      token_id=9,
      position=1,
      delta,
      source_offset=1,
      length=5,
    ),
    LunaEventPublishReady,
  )
  let token_view = owner.view()
  adapter.stage(token_view)
  luna_assert_adapter_matches(
    adapter,
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
  )
  adapter.release()
  owner.retire(token_view)
  let empty = FixedArray::make(0, b'\x00')
  assert_eq(
    owner.publish_token_status(
      token_id=2,
      position=0,
      empty,
      source_offset=0,
      length=0,
    ),
    LunaEventPublishReady,
  )
  let empty_view = owner.view()
  adapter.stage(empty_view)
  luna_assert_adapter_matches(
    adapter,
    Token(
      @inference.TokenEvent::new(
        token_id=2,
        decoded_delta=None,
        position=0,
        inference,
      ),
    ),
  )
  assert_eq(adapter.length(), EVENT_HEADER_BYTES)
  adapter.release()
  owner.retire(empty_view)
  let tail = FixedArray::make(inference.max_decoded_delta_bytes(), b't')
  owner.publish_completed(
    TokenLimit,
    input_tokens=3,
    cached_input_tokens=1,
    output_tokens=2,
    tail,
    tail_offset=0,
    tail_length=tail.length(),
  )
  let completed_view = owner.view()
  adapter.stage(completed_view)
  let usage = @inference.Usage::initial(
    input_tokens=3,
    cached_input_tokens=1,
    inference,
  ).advance(output_tokens=2, inference)
  luna_assert_adapter_matches(
    adapter,
    Completed(
      @inference.CompletedEvent::with_final_delta(
        finish_reason=TokenLimit,
        final_usage=usage,
        final_delta=@inference.DecodedDelta::from_utf8(
          Bytes::makei(tail.length(), _ => b't'),
          inference,
        ),
        inference,
      ),
    ),
  )
  adapter.release()
  owner.retire(completed_view)
  owner.publish_completed(
    StopSequence,
    input_tokens=3,
    cached_input_tokens=1,
    output_tokens=2,
    empty,
    tail_offset=0,
    tail_length=0,
  )
  let stop_view = owner.view()
  adapter.stage(stop_view)
  luna_assert_adapter_matches(
    adapter,
    Completed(
      @inference.CompletedEvent::new(
        finish_reason=StopSequence,
        final_usage=usage,
      ),
    ),
  )
  adapter.release()
  owner.retire(stop_view)
  owner.publish_completed(
    Cancelled,
    input_tokens=3,
    cached_input_tokens=1,
    output_tokens=2,
    empty,
    tail_offset=0,
    tail_length=0,
  )
  let cancelled_view = owner.view()
  adapter.stage(cancelled_view)
  luna_assert_adapter_matches(
    adapter,
    Completed(
      @inference.CompletedEvent::new(finish_reason=Cancelled, final_usage=usage),
    ),
  )
}

///|
test "Luna adapter busy stale and copy rejection preserve exact credit" {
  let inference = fixture_inference_limits()
  let owner = @luna_event.LunaEventOwner::new(inference)
  owner.publish_usage(input_tokens=2, cached_input_tokens=0, output_tokens=1)
  let view = owner.view()
  let adapter = LunaFramedEventAdapter::new(fixture_limits())
  adapter.stage(view)
  let pinned = luna_adapter_bytes(adapter)
  owner.retire(view)
  try adapter.stage(view) catch {
    EventCreditUnavailable => ()
    _ => fail("unexpected busy adapter error")
  } noraise {
    _ => fail("busy adapter replaced its pinned frame")
  }
  assert_fixed_bytes_equal(luna_adapter_bytes(adapter), pinned)
  let too_small = FixedArray::make(adapter.length() - 1, b'x')
  try adapter.copy_to(too_small, destination_offset=0) catch {
    DestinationTooSmall => ()
    _ => fail("unexpected adapter destination error")
  } noraise {
    _ => fail("small adapter destination passed")
  }
  assert_fixed_bytes_equal(luna_adapter_bytes(adapter), pinned)
  adapter.release()
  try adapter.stage(view) catch {
    StaleFrame => ()
    _ => fail("unexpected stale semantic view error")
  } noraise {
    _ => fail("retired semantic view staged")
  }
  try adapter.length() catch {
    StaleFrame => ()
    _ => fail("unexpected unpublished adapter error")
  } noraise {
    _ => fail("stale staging published a frame")
  }
  try adapter.release() catch {
    StaleFrame => ()
    _ => fail("unexpected duplicate release error")
  } noraise {
    _ => fail("duplicate adapter release passed")
  }
}

///|
test "Luna adapter capacity rejection is transactional and has no ACK" {
  let inference = fixture_inference_limits()
  let limits = fixture_limits()
  let adapter : LunaFramedEventAdapter = {
    limits,
    storage: FixedArray::make(REQUEST_HEADER_BYTES, b'\x00'),
    busy: false,
    length: 0,
  }
  let owner = @luna_event.LunaEventOwner::new(inference)
  let code = FixedArray::make(64, b'a')
  owner.publish_failed(code, code_offset=0, code_length=64, retryable=false)
  let oversized = owner.view()
  try adapter.stage(oversized) catch {
    FrameTooLarge(frame=EventFrame) => ()
    _ => fail("unexpected adapter capacity error")
  } noraise {
    _ => fail("oversized semantic event staged")
  }
  try adapter.length() catch {
    StaleFrame => ()
    _ => fail("unexpected adapter state after capacity rejection")
  } noraise {
    _ => fail("capacity rejection consumed adapter credit")
  }
  assert_eq(oversized.failed().code_length(), 64)
  owner.retire(oversized)
  owner.publish_usage(input_tokens=1, cached_input_tokens=0, output_tokens=0)
  let usage = owner.view()
  adapter.stage(usage)
  adapter.release()
  assert_eq(usage.usage().total_tokens(), 1)
  owner.retire(usage)
}

///|
fn luna_adapter_inference_limits(
  max_delta_bytes : Int,
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
    max_decoded_delta_bytes=max_delta_bytes,
    max_deadline_millis=10000UL,
    max_top_k=256,
    max_temperature=8.0,
  )
}

///|
fn luna_assert_adapter_limit_rejected(
  inference : @inference.InferenceLimits,
  max_frame_bytes : Int,
) -> Unit raise {
  let limits = FramedWireLimits::new(inference, max_frame_bytes~)
  try LunaFramedEventAdapter::new(limits) catch {
    InvalidLimit(FrameBytes) => ()
    _ => fail("unexpected Luna adapter envelope error")
  } noraise {
    _ => fail("undersized Luna adapter envelope passed")
  }
}

///|
test "Luna adapter constructor requires the exact semantic payload envelope" {
  let small_delta = luna_adapter_inference_limits(16)
  let public_code_envelope = EVENT_HEADER_BYTES + 64
  luna_assert_adapter_limit_rejected(small_delta, public_code_envelope - 1)
  ignore(
    LunaFramedEventAdapter::new(
      FramedWireLimits::new(small_delta, max_frame_bytes=public_code_envelope),
    ),
  )
  let large_delta = luna_adapter_inference_limits(80)
  let decoded_delta_envelope = EVENT_HEADER_BYTES + 80
  luna_assert_adapter_limit_rejected(large_delta, decoded_delta_envelope - 1)
  ignore(
    LunaFramedEventAdapter::new(
      FramedWireLimits::new(large_delta, max_frame_bytes=decoded_delta_envelope),
    ),
  )
}
