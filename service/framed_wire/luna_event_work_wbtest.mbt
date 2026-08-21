///|
fn luna_drive_event_work(
  semantic : @luna_event.LunaEventView,
  budget_units : Int,
) -> (FixedArray[Byte], UInt64) raise {
  let budget = LunaFramedEventStepBudget::new(budget_units)
  let workspace = LunaFramedEventWorkspace::new(fixture_limits(), budget)
  let work = workspace.begin(semantic)
  let mut summed = 0UL
  while work.state() == LunaFramedEventPending {
    assert_eq(work.progress(), LunaFramedEventAdvanced)
    let used = work.last_work_units()
    assert_true(used > 0)
    assert_true(used <= budget_units)
    summed += used.to_uint64()
  }
  assert_eq(work.state(), LunaFramedEventReady)
  assert_eq(work.total_work_units(), summed)
  assert_eq(work.progress(), LunaFramedEventIdle)
  assert_eq(work.last_work_units(), 0)
  let view = work.take_view()
  let output = FixedArray::make(view.length(), b'\x00')
  let mut cursor = 0
  while cursor < output.length() {
    let amount = Int::min(budget_units, output.length() - cursor)
    assert_eq(
      view.copy_chunk_to(
        output,
        destination_offset=cursor,
        source_offset=cursor,
        length=amount,
      ),
      amount,
    )
    cursor += amount
  }
  view.release()
  (output, summed)
}

///|
test "Luna framed event workspace sizing and budget bounds are exact" {
  let limits = fixture_limits()
  assert_eq(
    LunaFramedEventWorkspace::required_byte_cells(limits),
    (EVENT_HEADER_BYTES + 64).to_uint64(),
  )
  assert_eq(LunaFramedEventWorkspace::required_reference_cells(limits), 1UL)
  for invalid in [0, -1, LUNA_FRAMED_EVENT_MAX_STEP_WORK_UNITS + 1] {
    try LunaFramedEventStepBudget::new(invalid) catch {
      InvalidLimit(LunaFramedEventWorkUnits) => ()
      _ => fail("unexpected Luna framed event budget error")
    } noraise {
      _ => fail("invalid Luna framed event budget passed")
    }
  }
}

///|
test "Luna framed event work is canonical and deterministic across budgets" {
  let inference = fixture_inference_limits()
  let owner = @luna_event.LunaEventOwner::new(inference)
  let payload : FixedArray[Byte] = [b'o', b'k', 0xe4, 0xb8, 0xad]
  assert_eq(
    owner.publish_token_status(
      token_id=9,
      position=1,
      payload,
      source_offset=0,
      length=payload.length(),
    ),
    LunaEventPublishReady,
  )
  let semantic = owner.view()
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
  let mut expected_total = 0UL
  for budget in [1, 17, LUNA_FRAMED_EVENT_MAX_STEP_WORK_UNITS] {
    let (actual, total) = luna_drive_event_work(semantic, budget)
    assert_fixed_bytes_equal(actual, expected)
    assert_eq(total, 428UL)
    if expected_total == 0UL {
      expected_total = total
    } else {
      assert_eq(total, expected_total)
    }
  }
  owner.retire(semantic)
}

///|
test "Luna framed event View enforces chunks and stale auth before ranges" {
  let owner = @luna_event.LunaEventOwner::new(fixture_inference_limits())
  owner.publish_usage(input_tokens=2, cached_input_tokens=1, output_tokens=1)
  let semantic = owner.view()
  let workspace = LunaFramedEventWorkspace::new(
    fixture_limits(),
    LunaFramedEventStepBudget::new(17),
  )
  let work = workspace.begin(semantic)
  while work.state() == LunaFramedEventPending {
    ignore(work.progress())
  }
  assert_eq(work.total_work_units(), 418UL)
  let view = work.take_view()
  let destination = FixedArray::make(view.length(), b'\x00')
  for item in [(0, 0), (-1, 1), (0, 18), (view.length(), 1)] {
    let (source, length) = item
    try
      view.copy_chunk_to(
        destination,
        destination_offset=0,
        source_offset=source,
        length~,
      )
    catch {
      LunaFramedEventInvalidRange => ()
      _ => fail("unexpected Luna framed event source-range error")
    } noraise {
      _ => fail("invalid Luna framed event source range passed")
    }
  }
  try
    view.copy_chunk_to(
      destination,
      destination_offset=destination.length(),
      source_offset=0,
      length=1,
    )
  catch {
    DestinationTooSmall => ()
    _ => fail("unexpected Luna framed event destination error")
  } noraise {
    _ => fail("invalid Luna framed event destination passed")
  }
  view.release()
  try
    view.copy_chunk_to(
      destination,
      destination_offset=-1,
      source_offset=-1,
      length=0,
    )
  catch {
    LunaFramedEventStale => ()
    _ => fail("Luna framed event stale auth did not precede ranges")
  } noraise {
    _ => fail("stale Luna framed event View copied")
  }
  owner.retire(semantic)
}

///|
test "Luna framed event Work failure abort and detached Ready reuse exact owner" {
  let owner = @luna_event.LunaEventOwner::new(fixture_inference_limits())
  owner.publish_usage(input_tokens=1, cached_input_tokens=0, output_tokens=0)
  let first_semantic = owner.view()
  let workspace = LunaFramedEventWorkspace::new(
    fixture_limits(),
    LunaFramedEventStepBudget::new(1),
  )
  let failed = workspace.begin(first_semantic)
  assert_eq(failed.progress(), LunaFramedEventAdvanced)
  owner.retire(first_semantic)
  assert_eq(failed.progress(), LunaFramedEventAdvanced)
  assert_eq(failed.state(), LunaFramedEventFailed)
  assert_eq(failed.failure(), StaleFrame)
  failed.abort()
  try failed.state() catch {
    LunaFramedEventStale => ()
    _ => fail("unexpected stale failed-Work error")
  } noraise {
    _ => fail("aborted failed Work remained live")
  }
  owner.publish_usage(input_tokens=1, cached_input_tokens=0, output_tokens=0)
  let second_semantic = owner.view()
  let ready = workspace.begin(second_semantic)
  while ready.state() == LunaFramedEventPending {
    ignore(ready.progress())
  }
  // READY has detached immutable framed bytes from semantic authority.
  owner.retire(second_semantic)
  let detached = ready.take_view()
  let bytes = FixedArray::make(detached.length(), b'\x00')
  ignore(
    detached.copy_chunk_to(
      bytes,
      destination_offset=0,
      source_offset=0,
      length=1,
    ),
  )
  detached.release()
  owner.publish_usage(input_tokens=1, cached_input_tokens=0, output_tokens=0)
  let third_semantic = owner.view()
  let pending = workspace.begin(third_semantic)
  try workspace.begin(third_semantic) catch {
    LunaFramedEventBusy => ()
    _ => fail("unexpected Luna framed event busy error")
  } noraise {
    _ => fail("busy Luna framed event workspace began twice")
  }
  pending.abort()
  owner.retire(third_semantic)
}
