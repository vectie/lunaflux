///|
fn luna_http1_control_test_bytes(text : String) -> FixedArray[Byte] {
  let bytes = @utf8.encode(text)
  FixedArray::makei(bytes.length(), index => bytes[index])
}

///|
fn luna_http1_control_test_request(
  text : String,
) -> (LunaHttp1ControlRequest, LunaHttp1ControlRequestState) {
  let workspace = try! LunaHttp1ControlWorkspace::new(
    try! LunaHttp1ControlLimits::new(max_head_bytes=256),
  )
  let request = try! workspace.begin()
  let source = luna_http1_control_test_bytes(text)
  ignore(try! request.offer(source, source_offset=0, length=source.length()))
  (request, try! request.state())
}

///|
test "control HTTP accepts only exact bodyless observational routes" {
  for
    pair in [
      ("/healthz", LunaHttp1Health),
      ("/readyz", LunaHttp1Readiness),
      ("/metrics", LunaHttp1Metrics),
    ] {
    let (route, expected) = pair
    let (request, state) = luna_http1_control_test_request(
      "GET " +
      route +
      " HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n",
    )
    assert_eq(state, LunaHttp1ControlReady)
    let view = try! request.take_view()
    assert_eq(try! view.route(), expected)
    try! view.release()
  }
}

///|
test "control HTTP rejects hostile method path body and transfer forms" {
  for
    text in [
      "POST /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n", "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n",
      "GET /readyz?probe=1 HTTP/1.1\r\nHost: localhost\r\n\r\n", "GET /readyz HTTP/1.0\r\nHost: localhost\r\n\r\n",
      "GET /metrics?scope=model HTTP/1.1\r\nHost: localhost\r\n\r\n", "GET /readyz HTTP/1.1\r\nContent-Length: 1\r\n\r\nx",
      "GET /readyz HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n", "GET /readyz HTTP/1.1\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n",
      "GET /readyz HTTP/1.1\r\n folded: value\r\n\r\n",
    ] {
    let (request, state) = luna_http1_control_test_request(text)
    assert_eq(state, LunaHttp1ControlFailed)
    try! request.abort()
  }
}

///|
test "control metrics responses are exact bounded and payload-safe" {
  for
    tuple in [
      (Some(0), true, 200),
      (Some(86400000), false, 503),
      (None, false, 503),
    ] {
    let (duration, readiness, status) = tuple
    let response = try! LunaHttp1ControlMetricsResponse::new(
      duration,
      readiness~,
    )
    let output = FixedArray::make(response.length(), b'\x00')
    assert_eq(
      try! response.copy_chunk(
        output,
        destination_offset=0,
        source_offset=0,
        length=output.length(),
      ),
      output.length(),
    )
    let text = @utf8.decode(
      Bytes::makei(output.length(), index => output[index]),
    )
    assert_eq(response.status_code(), status)
    assert_true(text.contains("\"schema\":\"lunaflux.runtime.startup.v1\""))
    assert_true(text.contains("\"readiness\":" + readiness.to_string()))
    assert_true(text.contains("Cache-Control: no-store"))
    assert_false(text.contains("model"))
    assert_false(text.contains("path"))
    assert_false(text.contains("secret"))
  }
}

///|
test "control metrics response rejects false publication and overflow" {
  for duration in [Some(-1), Some(86400001)] {
    try LunaHttp1ControlMetricsResponse::new(duration, readiness=false) catch {
      LunaHttp1Failed(rule=LunaHttp1Limits, issue=LunaHttp1OutOfRange) => ()
      _ => fail("unexpected startup measurement range rejection")
    } noraise {
      _ => fail("expected startup measurement range rejection")
    }
  }
  try LunaHttp1ControlMetricsResponse::new(None, readiness=true) catch {
    LunaHttp1Failed(rule=LunaHttp1Lifecycle, issue=LunaHttp1Invalid) => ()
    _ => fail("unexpected absent-ready startup measurement rejection")
  } noraise {
    _ => fail("expected absent-ready startup measurement rejection")
  }
}

///|
test "control HTTP rejects incomplete and oversized heads without reuse debt" {
  let workspace = try! LunaHttp1ControlWorkspace::new(
    try! LunaHttp1ControlLimits::new(max_head_bytes=64),
  )
  let incomplete = try! workspace.begin()
  let prefix = luna_http1_control_test_bytes("GET /healthz HTTP/1.1\r\n")
  ignore(try! incomplete.offer(prefix, source_offset=0, length=prefix.length()))
  try! incomplete.finish_input()
  assert_eq(try! incomplete.state(), LunaHttp1ControlFailed)
  try! incomplete.abort()

  let oversized = try! workspace.begin()
  let source = FixedArray::make(65, b'A')
  assert_eq(
    try! oversized.offer(source, source_offset=0, length=source.length()),
    64,
  )
  assert_eq(try! oversized.state(), LunaHttp1ControlFailed)
  try! oversized.abort()

  let replacement = try! workspace.begin()
  let valid = luna_http1_control_test_bytes("GET /healthz HTTP/1.1\r\n\r\n")
  assert_eq(
    try! replacement.offer(valid, source_offset=0, length=valid.length()),
    valid.length(),
  )
  let view = try! replacement.take_view()
  assert_eq(try! view.route(), LunaHttp1Health)
  try! view.release()
}

///|
test "control responses are payload-safe fixed status documents" {
  for
    tuple in [
      (LunaHttp1ControlHealthy, 200, "{\"status\":\"healthy\"}\n"),
      (LunaHttp1ControlUnhealthy, 503, "{\"status\":\"unhealthy\"}\n"),
      (LunaHttp1ControlReadyResponse, 200, "{\"status\":\"ready\"}\n"),
      (LunaHttp1ControlNotReadyResponse, 503, "{\"status\":\"not_ready\"}\n"),
      (LunaHttp1ControlBadRequest, 400, "{\"status\":\"bad_request\"}\n"),
    ] {
    let (kind, code, body) = tuple
    let response = LunaHttp1ControlResponse::new(kind)
    let output = FixedArray::make(response.length(), b'\x00')
    assert_eq(
      try! response.copy_chunk(
        output,
        destination_offset=0,
        source_offset=0,
        length=output.length(),
      ),
      output.length(),
    )
    let text = @utf8.decode(
      Bytes::makei(output.length(), index => output[index]),
    )
    assert_eq(response.status_code(), code)
    assert_true(text.has_suffix(body))
    assert_true(text.contains("Cache-Control: no-store"))
    assert_true(text.contains("Connection: close"))
    assert_false(text.contains("secret"))
    assert_false(text.contains("path"))
  }
}
