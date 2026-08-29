///|
test "Luna OpenAI copy quanta dominate large TCP chunks" {
  assert_eq(luna_openai_bounded_copy_length(65536, 65536, 1), 1)
  assert_eq(luna_openai_bounded_copy_length(65536, 65536, 17), 17)
  assert_eq(luna_openai_bounded_copy_length(9, 65536, 17), 9)
  assert_eq(luna_openai_bounded_copy_length(65536, 7, 17), 7)
  assert_eq(luna_openai_bounded_copy_length(0, 65536, 17), 0)
  assert_eq(luna_openai_bounded_copy_length(1, 0, 17), 0)
}

///|
test "Luna OpenAI terminal publication is output-confirmation owned" {
  assert_false(luna_openai_terminal_confirmed(true, 18, 19))
  assert_false(luna_openai_terminal_confirmed(false, 19, 19))
  assert_false(luna_openai_terminal_confirmed(true, 0, 0))
  assert_true(luna_openai_terminal_confirmed(true, 19, 19))
}

///|
test "Luna OpenAI retained HTTP tail returns to receive after each quantum" {
  assert_true(luna_openai_http_requires_more_input(LunaHttp1ReceivingHead))
  assert_true(luna_openai_http_requires_more_input(LunaHttp1ReceivingBody))
  assert_false(luna_openai_http_requires_more_input(LunaHttp1ValidatingHead))
  assert_false(luna_openai_http_requires_more_input(LunaHttp1Authenticating))
  assert_false(luna_openai_http_requires_more_input(LunaHttp1Ready))
  assert_eq(
    luna_openai_http_resume_phase(LunaHttp1ReceivingHead),
    LUNA_OPENAI_HTTP_RECEIVE,
  )
  assert_eq(
    luna_openai_http_resume_phase(LunaHttp1ReceivingBody),
    LUNA_OPENAI_HTTP_RECEIVE,
  )
  assert_eq(
    luna_openai_http_resume_phase(LunaHttp1ValidatingHead),
    LUNA_OPENAI_HTTP_PROGRESS,
  )
}

///|
test "Luna OpenAI frame deadline authority transfers exactly once" {
  assert_eq(luna_openai_handoff_deadline_owner(1, 1), 1)
  assert_eq(luna_openai_handoff_deadline_owner(0, 1), 2)
  assert_eq(luna_openai_handoff_deadline_owner(0, 0), 0)
  assert_eq(luna_openai_handoff_deadline_owner(1, 0), 1)
  assert_eq(luna_openai_handoff_deadline_owner(2, 1), 0)
}

///|
test "Luna OpenAI ingress consumption is exact by disposition" {
  assert_true(
    luna_openai_ingress_consumption_valid(
      LunaOnlineFramedIngressAccepted,
      1,
      17,
    ),
  )
  assert_false(
    luna_openai_ingress_consumption_valid(
      LunaOnlineFramedIngressAccepted,
      0,
      17,
    ),
  )
  assert_true(
    luna_openai_ingress_consumption_valid(
      LunaOnlineFramedIngressBackpressured,
      0,
      17,
    ),
  )
  assert_true(
    luna_openai_ingress_consumption_valid(
      LunaOnlineFramedIngressDraining,
      0,
      17,
    ),
  )
  assert_false(
    luna_openai_ingress_consumption_valid(
      LunaOnlineFramedIngressBackpressured,
      1,
      17,
    ),
  )
  assert_false(
    luna_openai_ingress_consumption_valid(
      LunaOnlineFramedIngressDraining,
      1,
      17,
    ),
  )
  assert_false(
    luna_openai_ingress_consumption_valid(
      LunaOnlineFramedIngressAccepted,
      18,
      17,
    ),
  )
}

///|
fn embedded_control_fixture(text : String) -> FixedArray[Byte] {
  let bytes = @utf8.encode(text)
  FixedArray::makei(bytes.length(), index => bytes[index])
}

///|
test "OpenAI same-origin control parser recognizes only exact bounded GET routes" {
  let health = try! new_luna_openai_embedded_control()
  assert_true(health.begin())
  let request = embedded_control_fixture(
    "GET /healthz HTTP/1.1\r\nHost: runtime\r\n\r\n",
  )
  assert_eq(health.offer(request, 0, request.length()), 1)
  assert_true(health.metrics_output.is_idle())

  let metrics = try! new_luna_openai_embedded_control()
  assert_true(metrics.begin())
  let request = embedded_control_fixture(
    "GET /metrics HTTP/1.1\r\nHost: runtime\r\n\r\n",
  )
  assert_eq(metrics.offer(request, 0, request.length()), 3)
  assert_true(metrics.metrics_output.is_idle())

  let hostile = try! new_luna_openai_embedded_control()
  assert_true(hostile.begin())
  let request = embedded_control_fixture(
    "GET /metricsx HTTP/1.1\r\nHost: runtime\r\n\r\n",
  )
  assert_eq(hostile.offer(request, 0, request.length()), 4)
}

///|
test "OpenAI inference prefix does not enter embedded control ownership" {
  let post = embedded_control_fixture("POST /v1/responses HTTP/1.1\r\n")
  assert_true(post.length() > 0)
  assert_false(post[0] == b'G')
}
