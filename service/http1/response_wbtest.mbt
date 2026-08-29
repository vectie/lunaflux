///|
fn luna_http1_response_ready(
  work : LunaHttp1ResponseWork,
  budget : Int,
) -> LunaHttp1ResponseView raise {
  let mut turns = 0
  while (try! work.state()) != LunaHttp1ResponseReady && turns < 1000 {
    assert_eq(try! work.progress(), LunaHttp1ResponseAdvanced)
    assert_true((try! work.last_work_units()) <= budget)
    turns += 1
  }
  if (try! work.state()) != LunaHttp1ResponseReady {
    abort("Luna HTTP/1 response ready bound")
  }
  try! work.take_view()
}

///|
fn luna_http1_assert_response(
  view : LunaHttp1ResponseView,
  status : Int,
  expected : String,
) -> Unit raise {
  let encoded = @utf8.encode(expected)
  assert_eq(try! view.status_code(), status)
  assert_eq(try! view.length(), encoded.length())
  for index in 0..<encoded.length() {
    assert_eq(try! view.byte_at(index), encoded[index])
  }
}

///|
test "Luna HTTP/1 event-stream head is exact across budgets" {
  let expected = "HTTP/1.1 200 OK\r\n" +
    "Content-Type: text/event-stream\r\n" +
    "Cache-Control: no-cache\r\n" +
    "Connection: close\r\n\r\n"
  assert_eq(LunaHttp1ResponseWorkspace::required_byte_cells(), 167UL)
  for budget in [1, 17, LUNA_HTTP1_MAX_STEP_WORK_UNITS] {
    let workspace = LunaHttp1ResponseWorkspace::new(
      try! LunaHttp1ResponseStepBudget::new(budget),
    )
    let work = try! workspace.begin_event_stream()
    let view = luna_http1_response_ready(work, budget)
    luna_http1_assert_response(view, 200, expected)
    assert_eq(workspace.total_work_units, 97UL)
    try! view.release()
  }
}

///|
test "Luna HTTP/1 fixed JSON errors have exact status and length" {
  let cases = [
    (
      LunaHttp1AuthenticationError,
      401,
      "HTTP/1.1 401 Unauthorized\r\n" +
      "Content-Type: application/json\r\n" +
      "WWW-Authenticate: Bearer\r\n" +
      "Content-Length: 41\r\n" +
      "Connection: close\r\n\r\n" +
      "{\"error\":{\"type\":\"authentication_error\"}}",
    ),
    (
      LunaHttp1ValidationError,
      400,
      "HTTP/1.1 400 Bad Request\r\n" +
      "Content-Type: application/json\r\n" +
      "Content-Length: 37\r\n" +
      "Connection: close\r\n\r\n" +
      "{\"error\":{\"type\":\"validation_error\"}}",
    ),
    (
      LunaHttp1CapacityError,
      429,
      "HTTP/1.1 429 Too Many Requests\r\n" +
      "Content-Type: application/json\r\n" +
      "Content-Length: 35\r\n" +
      "Connection: close\r\n\r\n" +
      "{\"error\":{\"type\":\"capacity_error\"}}",
    ),
    (
      LunaHttp1ServiceError,
      503,
      "HTTP/1.1 503 Service Unavailable\r\n" +
      "Content-Type: application/json\r\n" +
      "Content-Length: 34\r\n" +
      "Connection: close\r\n\r\n" +
      "{\"error\":{\"type\":\"service_error\"}}",
    ),
  ]
  for item in cases {
    let (kind, status, expected) = item
    for budget in [1, 17, LUNA_HTTP1_MAX_STEP_WORK_UNITS] {
      let workspace = LunaHttp1ResponseWorkspace::new(
        try! LunaHttp1ResponseStepBudget::new(budget),
      )
      let work = try! workspace.begin_error(kind)
      let view = luna_http1_response_ready(work, budget)
      luna_http1_assert_response(view, status, expected)
      assert_eq(
        workspace.total_work_units,
        (try! view.length()).to_uint64() + 1UL,
      )
      try! view.release()
    }
  }
}

///|
test "Luna HTTP/1 response chunk copy is bounded and transactional" {
  let workspace = LunaHttp1ResponseWorkspace::new(
    try! LunaHttp1ResponseStepBudget::new(17),
  )
  let work = try! workspace.begin_event_stream()
  let view = luna_http1_response_ready(work, 17)
  let destination = FixedArray::make(34, b'!')
  assert_eq(
    try! view.copy_chunk(
      destination,
      destination_offset=2,
      source_offset=0,
      length=34,
    ),
    17,
  )
  assert_eq(destination[0], b'!')
  assert_eq(destination[2], b'H')
  assert_eq(destination[16], b'K')
  try
    view.copy_chunk(
      destination,
      destination_offset=30,
      source_offset=0,
      length=17,
    )
  catch {
    LunaHttp1Failed(rule=LunaHttp1Source, issue=LunaHttp1OutOfRange) => ()
    _ => fail("unexpected HTTP response copy-range error")
  } noraise {
    _ => fail("invalid HTTP response copy range was accepted")
  }
  assert_true(view.is_live())
  assert_eq(try! view.byte_at(0), b'H')
  try! view.release()
}

///|
test "Luna HTTP/1 response abort release stale and reuse are transactional" {
  let workspace = LunaHttp1ResponseWorkspace::new(
    try! LunaHttp1ResponseStepBudget::new(1),
  )
  let aborted = try! workspace.begin_event_stream()
  ignore(try! aborted.progress())
  try! aborted.abort()
  let first = try! workspace.begin_error(LunaHttp1ServiceError)
  let first_view = luna_http1_response_ready(first, 1)
  try first.state() catch {
    LunaHttp1Failed(rule=LunaHttp1Lifecycle, issue=LunaHttp1Stale) => ()
    _ => fail("unexpected HTTP response work stale error")
  } noraise {
    _ => fail("HTTP response Work stayed live after View lease")
  }
  try! first_view.release()
  assert_false(first_view.is_live())
  let second = try! workspace.begin_event_stream()
  try first_view.byte_at(0) catch {
    LunaHttp1Failed(rule=LunaHttp1Lifecycle, issue=LunaHttp1Stale) => ()
    _ => fail("unexpected HTTP response View stale error")
  } noraise {
    _ => fail("stale HTTP response View exposed a byte")
  }
  try! second.abort()
  workspace.epoch = 0xffffffffffffffffUL
  try workspace.begin_event_stream() catch {
    LunaHttp1Failed(rule=LunaHttp1Lifecycle, issue=LunaHttp1Exhausted) => ()
    _ => fail("unexpected HTTP response epoch error")
  } noraise {
    _ => fail("exhausted HTTP response epoch was reused")
  }
}

///|
test "Luna HTTP/1 response budgets busy state and stale-before-range are exact" {
  for invalid in [0, -1, LUNA_HTTP1_MAX_STEP_WORK_UNITS + 1] {
    try LunaHttp1ResponseStepBudget::new(invalid) catch {
      LunaHttp1Failed(rule=LunaHttp1Limits, issue=LunaHttp1OutOfRange) => ()
      _ => fail("unexpected HTTP response budget error")
    } noraise {
      _ => fail("invalid HTTP response budget was accepted")
    }
  }
  let workspace = LunaHttp1ResponseWorkspace::new(
    try! LunaHttp1ResponseStepBudget::new(1),
  )
  let work = try! workspace.begin_event_stream()
  try workspace.begin_error(LunaHttp1CapacityError) catch {
    LunaHttp1Failed(rule=LunaHttp1Lifecycle, issue=LunaHttp1Busy) => ()
    _ => fail("unexpected HTTP response busy error")
  } noraise {
    _ => fail("busy HTTP response workspace admitted a second Work")
  }
  let view = luna_http1_response_ready(work, 1)
  try view.byte_at(-1) catch {
    LunaHttp1Failed(rule=LunaHttp1Source, issue=LunaHttp1OutOfRange) => ()
    _ => fail("unexpected HTTP response byte bound error")
  } noraise {
    _ => fail("negative HTTP response byte index was accepted")
  }
  try! view.release()
  let destination = FixedArray::make(1, b'!')
  try
    view.copy_chunk(
      destination,
      destination_offset=-1,
      source_offset=-1,
      length=-1,
    )
  catch {
    LunaHttp1Failed(rule=LunaHttp1Lifecycle, issue=LunaHttp1Stale) => ()
    _ => fail("HTTP response copy did not authenticate stale first")
  } noraise {
    _ => fail("stale HTTP response copy was accepted")
  }
}
