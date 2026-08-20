///|
fn luna_token_source() -> (FixedArray[Byte], Int) raise {
  let inference = fixture_inference_limits()
  let request = @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(8UL),
    model_identity=fixture_identity(),
    input=@inference.Input::from_token_ids([1, 2, 3], inference),
    max_new_tokens=2,
    context_ceiling=8,
    sampling=@inference.SamplingParameters::greedy(),
    sampling_seed=@inference.SamplingSeed::new(3UL),
    stops=@inference.StopConditions::new([], [], inference),
    stream_preference=Buffered,
    deadline=@inference.DeadlineBudget::from_milliseconds(50UL, inference),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"scope", inference),
      Disabled,
    ),
    trace=None,
    limits=inference,
  )
  let frame = RequestFrameBuffer::new(fixture_limits()).encode(request)
  let source = FixedArray::make(frame.length(), b'\x00')
  frame.copy_to(source, destination_offset=0)
  (source, source.length())
}

///|
fn assert_luna_work_stale(work : LunaFramedRequestWork) -> Unit raise {
  try work.state() catch {
    LunaFramedRequestStale => ()
    _ => fail("stale work produced the wrong error")
  } noraise {
    _ => fail("stale work remained readable")
  }
  try work.progress() catch {
    LunaFramedRequestStale => ()
    _ => fail("stale work progress produced the wrong error")
  } noraise {
    _ => fail("stale work remained progressable")
  }
}

///|
fn assert_luna_view_stale(view : LunaFramedRequestView) -> Unit raise {
  try view.length() catch {
    LunaFramedRequestStale => ()
    _ => fail("stale view produced the wrong error")
  } noraise {
    _ => fail("stale view remained readable")
  }
  try view.release() catch {
    LunaFramedRequestStale => ()
    _ => fail("stale view release produced the wrong error")
  } noraise {
    _ => fail("stale view was released twice")
  }
}

///|
test "Luna framed request authenticates work view abort and reuse epochs" {
  let (source, length) = luna_request_source()
  let workspace = LunaFramedRequestWorkspace::new(
    fixture_limits(),
    LunaFramedRequestStepBudget::new(17),
  )
  let aborted = workspace.begin()
  try workspace.begin() catch {
    LunaFramedRequestBusy => ()
    _ => fail("live work produced the wrong busy error")
  } noraise {
    _ => fail("workspace admitted concurrent work")
  }
  assert_eq(aborted.offer(source, source_offset=0, length=0), 0)
  assert_eq(aborted.last_work_units(), 0)
  assert_eq(aborted.total_work_units(), 0UL)
  aborted.abort()
  assert_luna_work_stale(aborted)

  let ready = workspace.begin()
  luna_feed_range(ready, source, 0, length, 17)
  let view = luna_finish_valid(ready, 17)
  assert_luna_work_stale(ready)
  try workspace.begin() catch {
    LunaFramedRequestBusy => ()
    _ => fail("leased view produced the wrong busy error")
  } noraise {
    _ => fail("workspace reused while a view was live")
  }
  view.release()
  assert_luna_view_stale(view)

  let reused = workspace.begin()
  luna_feed_range(reused, source, 0, length, 17)
  while reused.state() is LunaFramedRequestValidating {
    luna_progress_once(reused, 17)
  }
  assert_eq(reused.state(), LunaFramedRequestReady)
  reused.abort()
  assert_luna_work_stale(reused)
}

///|
test "Luna framed request failure and invalid range charge zero then reuse" {
  let workspace = LunaFramedRequestWorkspace::new(
    fixture_limits(),
    LunaFramedRequestStepBudget::new(1),
  )
  let failed = workspace.begin()
  try failed.failure() catch {
    LunaFramedRequestNotReady => ()
    _ => fail("pending failure inspection produced the wrong error")
  } noraise {
    _ => fail("pending work exposed failure evidence")
  }
  try failed.offer([b'x'], source_offset=1, length=1) catch {
    InvalidFrame(frame=RequestFrame, rule=Length) => ()
    _ => fail("invalid offer range produced the wrong error")
  } noraise {
    _ => fail("invalid offer range was accepted")
  }
  assert_eq(failed.last_work_units(), 0)
  assert_eq(failed.total_work_units(), 0UL)
  assert_eq(failed.state(), LunaFramedRequestFailed)
  failed.abort()

  let (source, length) = luna_request_source()
  let fresh = workspace.begin()
  luna_feed_range(fresh, source, 0, length, 1)
  let view = luna_finish_valid(fresh, 1)
  assert_eq(view.request_id_value(), 42UL)
  view.release()
}

///|
test "Luna framed request single offer preserves following frame bytes" {
  let (source, length) = luna_request_source()
  let pipeline = FixedArray::make(length + 7, b'Z')
  for index in 0..<length {
    pipeline[index] = source[index]
  }
  let workspace = LunaFramedRequestWorkspace::new(
    fixture_limits(),
    LunaFramedRequestStepBudget::new(65536),
  )
  let work = workspace.begin()
  assert_eq(work.offer(pipeline, source_offset=0, length=pipeline.length()), 16)
  luna_progress_once(work, 65536)
  assert_eq(
    work.offer(pipeline, source_offset=16, length=pipeline.length() - 16),
    length - 16,
  )
  for index in length..<pipeline.length() {
    assert_eq(pipeline[index], b'Z')
  }
  let view = luna_finish_valid(work, 65536)
  assert_eq(view.length(), length)
  view.release()
}

///|
test "Luna framed request token view is scalar and bounds checked" {
  let (source, length) = luna_token_source()
  let (view, _) = luna_drive_split(source, length, 13, 1)
  assert_eq(view.input_kind(), LunaFramedRequestTokenIds)
  assert_eq(view.input_length(), 3)
  assert_eq(view.input_token_at(0), 1)
  assert_eq(view.input_token_at(1), 2)
  assert_eq(view.input_token_at(2), 3)
  assert_eq(view.sampling_mode(), Greedy)
  assert_false(view.has_top_k())
  assert_false(view.has_top_p())
  assert_eq(view.top_k(), 0)
  assert_eq(view.top_p(), 0.0)
  assert_eq(view.trace_length(), 0)
  try view.input_byte_at(0) catch {
    InvalidFrame(frame=RequestFrame, rule=Input) => ()
    _ => fail("wrong-kind byte read produced the wrong error")
  } noraise {
    _ => fail("token input exposed text bytes")
  }
  try view.input_token_at(3) catch {
    InvalidFrame(frame=RequestFrame, rule=Input) => ()
    _ => fail("token bound produced the wrong error")
  } noraise {
    _ => fail("token view accepted an out-of-range index")
  }
  view.release()
}
