///|
test "Luna HTTP/1 preserves every caller split and the pipelined tail" {
  let request = luna_http1_responses_request()
  for split in 0..<=request.length() {
    let workspace = luna_http1_test_workspace(17)
    let work = try! workspace.begin()
    let first = luna_http1_offer_request(work, request, offset=0, length=split)
    assert_eq(first, split)
    let second = luna_http1_offer_request(
      work,
      request,
      offset=split,
      length=request.length() - split,
    )
    assert_eq(first + second, request.length())
    let view = luna_http1_finish_ready(work)
    assert_eq(try! view.body_byte_at(0), b'{')
    try! view.release()
  }

  let first = luna_http1_responses_request()
  let second = luna_http1_chat_request_bytes()
  let combined = FixedArray::make(first.length() + second.length(), b'\x00')
  for index in 0..<first.length() {
    combined[index] = first[index]
  }
  for index in 0..<second.length() {
    combined[first.length() + index] = second[index]
  }
  let workspace = luna_http1_test_workspace(LUNA_HTTP1_MAX_STEP_WORK_UNITS)
  let work = try! workspace.begin()
  let consumed = luna_http1_offer_request(work, combined)
  assert_eq(consumed, first.length())
  let view = luna_http1_finish_ready(work)
  try! view.release()
  let next = try! workspace.begin()
  assert_eq(
    luna_http1_offer_request(next, combined, offset=consumed),
    second.length(),
  )
  let next_view = luna_http1_finish_ready(next)
  assert_eq(try! next_view.route(), LunaHttp1ChatCompletions)
  try! next_view.release()
}

///|
fn luna_http1_chat_request_bytes() -> FixedArray[Byte] {
  luna_http1_test_bytes(
    "POST /v1/chat/completions HTTP/1.1\r\n" +
    "Authorization: Bearer secret\r\n" +
    "Content-Length: 2\r\n" +
    "Content-Type: application/json\r\n\r\n{}",
  )
}

///|
test "Luna HTTP/1 work accounting is exact across budgets" {
  let request = luna_http1_responses_request()
  let mut expected = 0UL
  for budget in [1, 17, LUNA_HTTP1_MAX_STEP_WORK_UNITS] {
    let workspace = luna_http1_test_workspace(budget)
    let work = try! workspace.begin()
    assert_eq(luna_http1_offer_request(work, request), request.length())
    let view = luna_http1_finish_ready(work)
    let total = workspace.total_work_units
    if expected == 0UL {
      expected = total
    } else {
      assert_eq(total, expected)
    }
    assert_eq(total, 347UL)
    try! view.release()
  }
}

///|
test "Luna HTTP/1 source mutation cannot alter accepted body" {
  let request = luna_http1_responses_request(body="abc")
  let workspace = luna_http1_test_workspace(17)
  let work = try! workspace.begin()
  assert_eq(luna_http1_offer_request(work, request), request.length())
  let view = luna_http1_finish_ready(work)
  request[request.length() - 3] = b'z'
  request[request.length() - 2] = b'z'
  request[request.length() - 1] = b'z'
  assert_eq(try! view.body_byte_at(0), b'a')
  assert_eq(try! view.body_byte_at(1), b'b')
  assert_eq(try! view.body_byte_at(2), b'c')
  try! view.release()
}
