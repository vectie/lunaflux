///|
fn luna_http1_expect_text_failure(
  text : String,
  rule : LunaHttp1Rule,
  issue : LunaHttp1Issue,
) -> Unit raise {
  let workspace = luna_http1_test_workspace(17)
  let work = try! workspace.begin()
  let source = luna_http1_test_bytes(text)
  ignore(luna_http1_offer_request(work, source))
  luna_http1_expect_failure(work, rule, issue)
  try! work.abort()
  let replacement = try! workspace.begin()
  try! replacement.abort()
}

///|
test "Luna HTTP/1 rejects unsupported request and header forms" {
  luna_http1_expect_text_failure(
    "GET /v1/responses HTTP/1.1\r\nAuthorization: Bearer secret\r\nContent-Length: 0\r\nContent-Type: application/json\r\n\r\n",
    LunaHttp1Method,
    LunaHttp1Unsupported,
  )
  luna_http1_expect_text_failure(
    "POST /v1/unknown HTTP/1.1\r\nAuthorization: Bearer secret\r\nContent-Length: 0\r\nContent-Type: application/json\r\n\r\n",
    LunaHttp1RouteRule,
    LunaHttp1Unsupported,
  )
  luna_http1_expect_text_failure(
    "POST /v1/responses HTTP/1.0\r\nAuthorization: Bearer secret\r\nContent-Length: 0\r\nContent-Type: application/json\r\n\r\n",
    LunaHttp1Version,
    LunaHttp1Unsupported,
  )
  luna_http1_expect_text_failure(
    "POST /v1/responses HTTP/1.1\r\nAuthorization: Bearer secret\r\nContent-Length: 0\r\nContent-Type: text/plain\r\n\r\n",
    LunaHttp1ContentType,
    LunaHttp1Unsupported,
  )
  luna_http1_expect_text_failure(
    "POST /v1/responses HTTP/1.1\r\nAuthorization: Bearer secret\r\nContent-Length: 0\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n",
    LunaHttp1TransferEncoding,
    LunaHttp1Unsupported,
  )
}

///|
test "Luna HTTP/1 rejects missing duplicate and conflicting authority headers" {
  luna_http1_expect_text_failure(
    "POST /v1/responses HTTP/1.1\r\nContent-Length: 0\r\nContent-Type: application/json\r\n\r\n",
    LunaHttp1Authorization,
    LunaHttp1Missing,
  )
  luna_http1_expect_text_failure(
    "POST /v1/responses HTTP/1.1\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\n\r\n",
    LunaHttp1ContentLength,
    LunaHttp1Missing,
  )
  luna_http1_expect_text_failure(
    "POST /v1/responses HTTP/1.1\r\nAuthorization: Bearer secret\r\nContent-Length: 0\r\n\r\n",
    LunaHttp1ContentType,
    LunaHttp1Missing,
  )
  luna_http1_expect_text_failure(
    "POST /v1/responses HTTP/1.1\r\nAuthorization: Bearer secret\r\nAuthorization: Bearer secret\r\nContent-Length: 0\r\nContent-Type: application/json\r\n\r\n",
    LunaHttp1Authorization,
    LunaHttp1Duplicate,
  )
  luna_http1_expect_text_failure(
    "POST /v1/responses HTTP/1.1\r\nAuthorization: Bearer secret\r\nContent-Length: 0\r\nContent-Length: 0\r\nContent-Type: application/json\r\n\r\n",
    LunaHttp1ContentLength,
    LunaHttp1Duplicate,
  )
  luna_http1_expect_text_failure(
    "POST /v1/responses HTTP/1.1\r\nAuthorization: Bearer secret\r\nContent-Length: 0\r\nContent-Length: 1\r\nContent-Type: application/json\r\n\r\nx",
    LunaHttp1ContentLength,
    LunaHttp1Conflicting,
  )
  luna_http1_expect_text_failure(
    "POST /v1/responses HTTP/1.1\r\nAuthorization: Bearer secret\r\nContent-Length: 0\r\nContent-Type: application/json\r\nContent-Type: application/json\r\n\r\n",
    LunaHttp1ContentType,
    LunaHttp1Duplicate,
  )
}

///|
test "Luna HTTP/1 rejects obs-fold controls and malformed Bearer syntax" {
  luna_http1_expect_text_failure(
    "POST /v1/responses HTTP/1.1\r\nAuthorization: Bearer secret\r\n Content-Length: 0\r\nContent-Type: application/json\r\n\r\n",
    LunaHttp1Header,
    LunaHttp1Invalid,
  )
  luna_http1_expect_text_failure(
    "POST /v1/responses HTTP/1.1\r\nAuthorization: Bearer sec\u0000ret\r\nContent-Length: 0\r\nContent-Type: application/json\r\n\r\n",
    LunaHttp1Header,
    LunaHttp1Invalid,
  )
  luna_http1_expect_text_failure(
    "POST /v1/responses HTTP/1.1\r\nAuthorization: Basic secret\r\nContent-Length: 0\r\nContent-Type: application/json\r\n\r\n",
    LunaHttp1Authorization,
    LunaHttp1Invalid,
  )
  luna_http1_expect_text_failure(
    "POST /v1/responses HTTP/1.1\r\nAuthorization: Bearer ab=c\r\nContent-Length: 0\r\nContent-Type: application/json\r\n\r\n",
    LunaHttp1Authorization,
    LunaHttp1Invalid,
  )
}

///|
test "Luna HTTP/1 authentication rejection consumes no body byte" {
  let workspace = luna_http1_test_workspace(17)
  let work = try! workspace.begin()
  let request = luna_http1_responses_request(credential="wrong", body="body")
  let consumed = luna_http1_offer_request(work, request)
  luna_http1_expect_failure(work, LunaHttp1Authorization, LunaHttp1Rejected)
  assert_eq(consumed, workspace.head_length)
  assert_eq(workspace.body_written, 0)
  assert_eq(workspace.storage[workspace.limits.max_head_bytes], b'\x00')
  try! work.abort()
  let replacement = try! workspace.begin()
  let valid = luna_http1_responses_request(body="ok")
  assert_eq(luna_http1_offer_request(replacement, valid), valid.length())
  let view = luna_http1_finish_ready(replacement)
  assert_eq(try! view.body_byte_at(0), b'o')
  try! view.release()
}

///|
test "Luna HTTP/1 checks every credential mismatch before body receipt" {
  for
    credential in [
      "s", "secretx", "aecret", "sacret", "searet", "secset", "secrat", "secreu",
    ] {
    let workspace = luna_http1_test_workspace(1)
    let work = try! workspace.begin()
    let request = luna_http1_responses_request(credential~, body="payload")
    let consumed = luna_http1_offer_request(work, request)
    luna_http1_expect_failure(work, LunaHttp1Authorization, LunaHttp1Rejected)
    assert_eq(consumed, workspace.head_length)
    assert_eq(workspace.body_written, 0)
    try! work.abort()
  }
  luna_http1_expect_text_failure(
    "POST /v1/responses HTTP/1.1\r\nAuthorization: Bearer \r\nContent-Length: 1\r\nContent-Type: application/json\r\n\r\nx",
    LunaHttp1Authorization,
    LunaHttp1Invalid,
  )
}

///|
test "Luna HTTP/1 head completion stops before authentication and body copy" {
  let workspace = luna_http1_test_workspace(LUNA_HTTP1_MAX_STEP_WORK_UNITS)
  let work = try! workspace.begin()
  let request = luna_http1_responses_request(body="body")
  let head_length = request.length() - 4
  assert_eq(
    try! work.offer(request, source_offset=0, length=request.length()),
    head_length,
  )
  assert_eq(workspace.head_length, head_length)
  assert_eq(workspace.body_written, 0)
  assert_eq(try! work.state(), LunaHttp1ValidatingHead)
  while (try! work.state()) != LunaHttp1ReceivingBody {
    ignore(try! work.progress())
  }
  assert_eq(workspace.body_written, 0)
  assert_eq(try! work.offer(request, source_offset=head_length, length=4), 4)
  let view = luna_http1_finish_ready(work)
  assert_eq(try! view.body_byte_at(0), b'b')
  try! view.release()
}

///|
test "Luna HTTP/1 source-range errors are transactional" {
  let workspace = luna_http1_test_workspace(1)
  let work = try! workspace.begin()
  let request = luna_http1_responses_request()
  try work.offer(request, source_offset=-1, length=1) catch {
    LunaHttp1Failed(rule=LunaHttp1Source, issue=LunaHttp1OutOfRange) => ()
    _ => fail("unexpected HTTP source-range error")
  } noraise {
    _ => fail("invalid HTTP source range was accepted")
  }
  assert_eq(try! work.state(), LunaHttp1ReceivingHead)
  assert_eq(workspace.head_length, 0)
  assert_eq(luna_http1_offer_request(work, request), request.length())
  let view = luna_http1_finish_ready(work)
  try! view.release()
}

///|
test "Luna HTTP/1 premature EOF and trailing offer are terminal" {
  let head_workspace = luna_http1_test_workspace(17)
  let head_work = try! head_workspace.begin()
  let request = luna_http1_responses_request(body="abcd")
  assert_eq(luna_http1_offer_request(head_work, request, length=10), 10)
  try! head_work.finish_input()
  luna_http1_expect_failure(head_work, LunaHttp1Head, LunaHttp1PrematureEnd)

  let body_workspace = luna_http1_test_workspace(17)
  let body_work = try! body_workspace.begin()
  assert_eq(
    luna_http1_offer_request(body_work, request, length=request.length() - 1),
    request.length() - 1,
  )
  assert_eq(try! body_work.state(), LunaHttp1ReceivingBody)
  try! body_work.finish_input()
  luna_http1_expect_failure(body_work, LunaHttp1Body, LunaHttp1PrematureEnd)

  let trailing_workspace = luna_http1_test_workspace(17)
  let trailing_work = try! trailing_workspace.begin()
  assert_eq(luna_http1_offer_request(trailing_work, request), request.length())
  while (try! trailing_work.state()) != LunaHttp1Ready {
    ignore(try! trailing_work.progress())
  }
  let extra : FixedArray[Byte] = FixedArray::make(1, b'x')
  try trailing_work.offer(extra, source_offset=0, length=1) catch {
    LunaHttp1Failed(rule=LunaHttp1Body, issue=LunaHttp1Trailing) => ()
    _ => fail("unexpected HTTP trailing-byte error")
  } noraise {
    _ => fail("trailing HTTP byte was accepted")
  }
  luna_http1_expect_failure(trailing_work, LunaHttp1Body, LunaHttp1Trailing)
}

///|
test "Luna HTTP/1 checked storage limits reject forged and overflowing shapes" {
  for head in [0, LUNA_HTTP1_MIN_HEAD_BYTES - 1, LUNA_HTTP1_MAX_HEAD_BYTES + 1] {
    try LunaHttp1Limits::new(max_head_bytes=head, max_body_bytes=0) catch {
      LunaHttp1Failed(rule=LunaHttp1Limits, issue=LunaHttp1OutOfRange) => ()
      _ => fail("unexpected HTTP head limit error")
    } noraise {
      _ => fail("invalid HTTP head limit was accepted")
    }
  }
  for body in [-1, LUNA_HTTP1_MAX_BODY_BYTES + 1] {
    try
      LunaHttp1Limits::new(
        max_head_bytes=LUNA_HTTP1_MIN_HEAD_BYTES,
        max_body_bytes=body,
      )
    catch {
      LunaHttp1Failed(rule=LunaHttp1Limits, issue=LunaHttp1OutOfRange) => ()
      _ => fail("unexpected HTTP body limit error")
    } noraise {
      _ => fail("invalid HTTP body limit was accepted")
    }
  }
  let maximum = try! LunaHttp1Limits::new(
    max_head_bytes=LUNA_HTTP1_MAX_HEAD_BYTES,
    max_body_bytes=LUNA_HTTP1_MAX_BODY_BYTES,
  )
  assert_eq(try! LunaHttp1Workspace::required_byte_cells(maximum), 16842752UL)
  let forged : LunaHttp1Limits = {
    max_head_bytes: -2147483647,
    max_body_bytes: 2147483647,
  }
  try LunaHttp1Workspace::required_byte_cells(forged) catch {
    LunaHttp1Failed(rule=LunaHttp1Limits, issue=LunaHttp1Invalid) => ()
    _ => fail("unexpected forged HTTP limit error")
  } noraise {
    _ => fail("forged HTTP limits were accepted")
  }

  let maximum_workspace = try! LunaHttp1Workspace::new(
    maximum,
    try! LunaHttp1StepBudget::new(LUNA_HTTP1_MAX_STEP_WORK_UNITS),
    luna_http1_test_policy(),
  )
  let maximum_work = try! maximum_workspace.begin()
  try! maximum_work.abort()

  for invalid_budget in [0, -1, LUNA_HTTP1_MAX_STEP_WORK_UNITS + 1] {
    try LunaHttp1StepBudget::new(invalid_budget) catch {
      LunaHttp1Failed(rule=LunaHttp1Limits, issue=LunaHttp1OutOfRange) => ()
      _ => fail("unexpected HTTP step-budget error")
    } noraise {
      _ => fail("invalid HTTP step budget was accepted")
    }
  }
}

///|
test "Luna HTTP/1 oversized head and declared body fail without overflow" {
  let head_workspace = luna_http1_test_workspace(
    17,
    max_head_bytes=LUNA_HTTP1_MIN_HEAD_BYTES,
  )
  let head_work = try! head_workspace.begin()
  let head = FixedArray::make(LUNA_HTTP1_MIN_HEAD_BYTES + 1, b'A')
  assert_eq(
    luna_http1_offer_request(head_work, head),
    LUNA_HTTP1_MIN_HEAD_BYTES,
  )
  luna_http1_expect_failure(head_work, LunaHttp1Head, LunaHttp1TooLong)

  luna_http1_expect_text_failure(
    "POST /v1/responses HTTP/1.1\r\nAuthorization: Bearer secret\r\nContent-Length: 1025\r\nContent-Type: application/json\r\n\r\n",
    LunaHttp1ContentLength,
    LunaHttp1TooLong,
  )
}

///|
test "Luna HTTP/1 work and view aliases become stale across generations" {
  let workspace = luna_http1_test_workspace(17)
  let request = luna_http1_responses_request()
  let first = try! workspace.begin()
  assert_eq(luna_http1_offer_request(first, request), request.length())
  let view = luna_http1_finish_ready(first)
  try first.state() catch {
    LunaHttp1Failed(rule=LunaHttp1Lifecycle, issue=LunaHttp1Stale) => ()
    _ => fail("unexpected leased-work stale error")
  } noraise {
    _ => fail("work remained live after view lease")
  }
  try! view.release()
  let second = try! workspace.begin()
  try first.abort() catch {
    LunaHttp1Failed(rule=LunaHttp1Lifecycle, issue=LunaHttp1Stale) => ()
    _ => fail("unexpected old-work stale error")
  } noraise {
    _ => fail("old work aborted a new generation")
  }
  try! second.abort()
  workspace.epoch = 0xffffffffffffffffUL
  try workspace.begin() catch {
    LunaHttp1Failed(rule=LunaHttp1Lifecycle, issue=LunaHttp1Exhausted) => ()
    _ => fail("unexpected HTTP epoch exhaustion error")
  } noraise {
    _ => fail("exhausted HTTP epoch was reused")
  }
}
