///|
fn luna_request_source() -> (FixedArray[Byte], Int) raise {
  let frame = RequestFrameBuffer::new(fixture_limits()).encode(
    fixture_request(),
  )
  let source = FixedArray::make(frame.length(), b'\x00')
  frame.copy_to(source, destination_offset=0)
  (source, source.length())
}

///|
fn assert_luna_work_charge(
  work : LunaFramedRequestWork,
  budget : Int,
) -> Unit raise {
  let used = work.last_work_units()
  assert_true(used > 0)
  assert_true(used <= budget)
}

///|
fn luna_progress_once(work : LunaFramedRequestWork, budget : Int) -> Unit raise {
  assert_eq(work.progress(), LunaFramedRequestAdvanced)
  assert_luna_work_charge(work, budget)
}

///|
fn luna_feed_range(
  work : LunaFramedRequestWork,
  source : FixedArray[Byte],
  start : Int,
  end : Int,
  budget : Int,
) -> Unit raise {
  let mut cursor = start
  while cursor < end {
    match work.state() {
      LunaFramedRequestValidating => luna_progress_once(work, budget)
      LunaFramedRequestReceivingPrefix | LunaFramedRequestReceivingBody => {
        let consumed = work.offer(
          source,
          source_offset=cursor,
          length=end - cursor,
        )
        assert_true(consumed > 0)
        assert_true(consumed <= end - cursor)
        assert_luna_work_charge(work, budget)
        cursor += consumed
      }
      LunaFramedRequestReady =>
        fail("request became ready before split input ended")
      LunaFramedRequestFailed => fail("valid request failed while receiving")
    }
  }
}

///|
fn luna_finish_valid(
  work : LunaFramedRequestWork,
  budget : Int,
) -> LunaFramedRequestView raise {
  while work.state() is LunaFramedRequestValidating {
    luna_progress_once(work, budget)
  }
  assert_eq(work.state(), LunaFramedRequestReady)
  work.take_view()
}

///|
fn luna_drive_split(
  source : FixedArray[Byte],
  length : Int,
  split : Int,
  budget : Int,
) -> (LunaFramedRequestView, UInt64) raise {
  let workspace = LunaFramedRequestWorkspace::new(
    fixture_limits(),
    LunaFramedRequestStepBudget::new(budget),
  )
  let work = workspace.begin()
  luna_feed_range(work, source, 0, split, budget)
  assert_eq(work.offer(source, source_offset=split, length=0), 0)
  assert_eq(work.last_work_units(), 0)
  luna_feed_range(work, source, split, length, budget)
  while work.state() is LunaFramedRequestValidating {
    luna_progress_once(work, budget)
  }
  assert_eq(work.state(), LunaFramedRequestReady)
  let total = work.total_work_units()
  (work.take_view(), total)
}

///|
fn assert_luna_text_view(view : LunaFramedRequestView) -> Unit raise {
  assert_eq(view.inference_limits(), fixture_inference_limits())
  assert_eq(view.length(), 304)
  assert_eq(view.protocol_version_wire(), 1)
  assert_eq(view.request_id_value(), 42UL)
  assert_eq(view.input_kind(), LunaFramedRequestText)
  let expected_input = b"hello moon"
  assert_eq(view.input_length(), expected_input.length())
  for index, byte in expected_input {
    assert_eq(view.input_byte_at(index), byte)
  }
  assert_eq(view.max_new_tokens(), 4)
  assert_eq(view.context_ceiling(), 16)
  assert_eq(view.sampling_mode(), Sample)
  assert_eq(view.sampling_temperature(), 0.75)
  assert_true(view.has_top_k())
  assert_eq(view.top_k(), 12)
  assert_true(view.has_top_p())
  assert_eq(view.top_p(), 0.9)
  assert_eq(view.sampling_seed_value(), 99UL)
  assert_eq(view.stream_preference(), Streaming)
  assert_eq(view.deadline_millis(), 5000UL)
  assert_eq(view.cache_permission(), ReadWrite)
  let content = b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  let plan = b"abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
  for index in 0..<64 {
    assert_eq(view.content_digest_byte_at(index), content[index])
    assert_eq(view.plan_digest_byte_at(index), plan[index])
  }
  assert_eq(view.stop_token_count(), 2)
  assert_eq(view.stop_token_at(0), 2)
  assert_eq(view.stop_token_at(1), 7)
  assert_eq(view.stop_string_count(), 2)
  for pair in [(0, b"END"), (1, b"stop")] {
    let (string_index, expected) = pair
    assert_eq(view.stop_string_length(string_index), expected.length())
    for byte_index, byte in expected {
      assert_eq(view.stop_string_byte_at(string_index, byte_index), byte)
    }
  }
  for pair in [(b"tenant-a", true), (b"trace-7", false)] {
    let (expected, scope) = pair
    let actual_length = if scope {
      view.cache_scope_length()
    } else {
      view.trace_length()
    }
    assert_eq(actual_length, expected.length())
    for index, byte in expected {
      let actual = if scope {
        view.cache_scope_byte_at(index)
      } else {
        view.trace_byte_at(index)
      }
      assert_eq(actual, byte)
    }
  }
}

///|
test "Luna framed request matches frozen text golden at every split and budget" {
  let (source, length) = luna_request_source()
  assert_eq(length, 304)
  assert_eq(
    @crypto.bytes_to_hex_string(
      @crypto.sha256(Bytes::makei(length, index => source[index])),
    ),
    "09e534bb3c7d1b9200362b23239dc5c11352ae50811313c800bbe6c717efa72f",
  )
  let mut expected_total = 0UL
  for budget in [1, 17, 65536] {
    for split in 0..<=length {
      let (view, total) = luna_drive_split(source, length, split, budget)
      assert_luna_text_view(view)
      if expected_total == 0UL {
        expected_total = total
      } else {
        assert_eq(total, expected_total)
      }
      view.release()
    }
  }
}

///|
test "Luna framed request preserves pipeline tail for explicit reuse" {
  let (source, length) = luna_request_source()
  let pipeline = FixedArray::make(length * 2, b'\x00')
  for index in 0..<length {
    pipeline[index] = source[index]
    pipeline[length + index] = source[index]
  }
  let workspace = LunaFramedRequestWorkspace::new(
    fixture_limits(),
    LunaFramedRequestStepBudget::new(17),
  )
  let first = workspace.begin()
  luna_feed_range(first, pipeline, 0, length, 17)
  let first_view = luna_finish_valid(first, 17)
  assert_luna_text_view(first_view)
  first_view.release()
  let second = workspace.begin()
  luna_feed_range(second, pipeline, length, length * 2, 17)
  let second_view = luna_finish_valid(second, 17)
  assert_luna_text_view(second_view)
  second_view.release()
}

///|
test "Luna framed request budget bounds are exact" {
  assert_eq(LunaFramedRequestStepBudget::new(1).work_units(), 1)
  assert_eq(LunaFramedRequestStepBudget::new(65536).work_units(), 65536)
  for invalid in [0, -1, 65537] {
    try LunaFramedRequestStepBudget::new(invalid) catch {
      InvalidLimit(LunaFramedRequestWorkUnits) => ()
      _ => fail("unexpected Luna request budget error")
    } noraise {
      _ => fail("invalid Luna request budget accepted")
    }
  }
}
