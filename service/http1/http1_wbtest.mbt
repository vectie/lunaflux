///|
fn luna_http1_test_bytes(text : String) -> FixedArray[Byte] {
  let encoded = @utf8.encode(text)
  FixedArray::makei(encoded.length(), index => encoded[index])
}

///|
fn luna_http1_test_policy() -> @api_auth.LunaApiAuthPolicy {
  let expected = luna_http1_test_bytes("secret")
  try! @api_auth.LunaApiAuthPolicy::new(
    expected,
    source_offset=0,
    length=expected.length(),
    maximum_credential_bytes=8,
  )
}

///|
fn luna_http1_test_workspace(
  budget : Int,
  max_head_bytes? : Int = 1024,
  max_body_bytes? : Int = 1024,
) -> LunaHttp1Workspace {
  let limits = try! LunaHttp1Limits::new(max_head_bytes~, max_body_bytes~)
  try! LunaHttp1Workspace::new(
    limits,
    try! LunaHttp1StepBudget::new(budget),
    luna_http1_test_policy(),
  )
}

///|
fn luna_http1_responses_request(
  credential? : String = "secret",
  body? : String = "{}",
) -> FixedArray[Byte] {
  luna_http1_test_bytes(
    "POST /v1/responses HTTP/1.1\r\n" +
    "Authorization: Bearer " +
    credential +
    "\r\n" +
    "Content-Length: " +
    body.length().to_string() +
    "\r\n" +
    "Content-Type: application/json\r\n\r\n" +
    body,
  )
}

///|
fn luna_http1_drive_until_receiving_or_terminal(work : LunaHttp1Work) -> Unit {
  let mut turns = 0
  while turns < 100000 {
    match (try! work.state()) {
      LunaHttp1ReceivingHead
      | LunaHttp1ReceivingBody
      | LunaHttp1Ready
      | LunaHttp1Failed => return
      _ => ignore(try! work.progress())
    }
    turns += 1
  }
  abort("Luna HTTP/1 test progress bound")
}

///|
fn luna_http1_offer_request(
  work : LunaHttp1Work,
  source : FixedArray[Byte],
  offset? : Int = 0,
  length? : Int = -1,
) -> Int raise {
  let target = if length < 0 { source.length() - offset } else { length }
  let mut consumed = 0
  let mut turns = 0
  while consumed < target && turns < 100000 {
    let amount = try! work.offer(
      source,
      source_offset=offset + consumed,
      length=target - consumed,
    )
    assert_true(
      (try! work.last_work_units()) <= work.workspace.budget.work_units(),
    )
    consumed += amount
    match (try! work.state()) {
      LunaHttp1Ready | LunaHttp1Failed => return consumed
      _ => ()
    }
    if amount == 0 {
      luna_http1_drive_until_receiving_or_terminal(work)
      match (try! work.state()) {
        LunaHttp1Ready | LunaHttp1Failed => return consumed
        _ => ()
      }
    }
    turns += 1
  }
  luna_http1_drive_until_receiving_or_terminal(work)
  consumed
}

///|
fn luna_http1_finish_ready(work : LunaHttp1Work) -> LunaHttp1View raise {
  let mut turns = 0
  while (try! work.state()) != LunaHttp1Ready && turns < 100000 {
    ignore(try! work.progress())
    assert_true(
      (try! work.last_work_units()) <= work.workspace.budget.work_units(),
    )
    turns += 1
  }
  if (try! work.state()) != LunaHttp1Ready {
    abort("Luna HTTP/1 ready bound")
  }
  try! work.take_view()
}

///|
fn luna_http1_expect_failure(
  work : LunaHttp1Work,
  rule : LunaHttp1Rule,
  issue : LunaHttp1Issue,
) -> Unit raise {
  assert_eq(try! work.state(), LunaHttp1Failed)
  match (try! work.failure()) {
    LunaHttp1Failed(rule=actual_rule, issue=actual_issue) => {
      assert_eq(actual_rule, rule)
      assert_eq(actual_issue, issue)
    }
  }
}

///|
test "Luna HTTP/1 authenticates before body receipt and leases immutable body" {
  let workspace = luna_http1_test_workspace(1)
  let work = try! workspace.begin()
  let request = luna_http1_responses_request(body="{\"x\":1}")
  let consumed = luna_http1_offer_request(work, request)
  assert_eq(consumed, request.length())
  let view = luna_http1_finish_ready(work)
  assert_true(view.is_live())
  assert_eq(try! view.route(), LunaHttp1Responses)
  assert_eq(try! view.body_length(), 7)
  assert_eq(try! view.body_byte_at(0), b'{')
  assert_eq(try! view.body_byte_at(6), b'}')
  try! view.release()
  assert_false(view.is_live())
  try view.body_byte_at(0) catch {
    LunaHttp1Failed(rule=LunaHttp1Lifecycle, issue=LunaHttp1Stale) => ()
    _ => fail("unexpected stale HTTP view error")
  } noraise {
    _ => fail("stale HTTP view exposed a body byte")
  }
}

///|
test "Luna HTTP/1 wipes credentials eagerly and full storage only on close" {
  let workspace = luna_http1_test_workspace(
    17,
    max_head_bytes=256,
    max_body_bytes=32,
  )
  let request = luna_http1_responses_request(body="{\"secret_body\":1}")
  let first = try! workspace.begin()
  assert_eq(luna_http1_offer_request(first, request), request.length())
  let first_view = luna_http1_finish_ready(first)
  assert_true(workspace.storage[0] != b'\x00')
  assert_true(workspace.storage[workspace.limits.max_head_bytes] != b'\x00')
  for index in 0..<6 {
    assert_eq(workspace.storage[workspace.credential_offset + index], b'\x00')
  }
  try! first_view.release()

  let second = try! workspace.begin()
  assert_true(workspace.storage[0] != b'\x00')
  assert_true(workspace.storage[workspace.limits.max_head_bytes] != b'\x00')
  try! second.abort()

  let closing = try! workspace.begin()
  assert_eq(luna_http1_offer_request(closing, request), request.length())
  assert_eq(try! closing.state(), LunaHttp1Ready)
  workspace.close_authentication()
  workspace.close_authentication()
  for byte in workspace.storage {
    assert_eq(byte, b'\x00')
  }
  try closing.state() catch {
    LunaHttp1Failed(rule=LunaHttp1Lifecycle, issue=LunaHttp1Stale) => ()
    _ => fail("unexpected closed HTTP work rejection")
  } noraise {
    _ => fail("HTTP work survived authentication close")
  }
}

///|
test "Luna HTTP/1 accepts chat route case-insensitive header names" {
  let workspace = luna_http1_test_workspace(17)
  let request = luna_http1_test_bytes(
    "POST /v1/chat/completions HTTP/1.1\r\n" +
    "aUtHoRiZaTiOn: Bearer secret\r\n" +
    "cOnTeNt-LeNgTh: 2\r\n" +
    "CONTENT-TYPE: APPLICATION/JSON\r\n\r\n{}",
  )
  let work = try! workspace.begin()
  assert_eq(luna_http1_offer_request(work, request), request.length())
  let view = luna_http1_finish_ready(work)
  assert_eq(try! view.route(), LunaHttp1ChatCompletions)
  try! view.release()
}
