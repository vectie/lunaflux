///|
test "normalized text admission retains original authenticated HTTP anchor and releases it" {
  let limits = openai_wb_fixture_limits()
  let workspace = inbound_workspace(limits, 1, max_messages=2)
  let http = inbound_http_view(2, "original media body")
  let body = @utf8.encode(
    "{\"model\":\"model-x\",\"stream\":false,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"normalized image placeholders\"}]}",
  )
  let work = workspace.begin_normalized(
    http,
    @inference.RequestId::new(9UL),
    body,
  )
  while work.state() is LunaOpenAIInboundPending {
    ignore(work.progress())
  }
  assert_eq(work.state(), LunaOpenAIInboundReady)
  assert_true(!http.is_live())
  let view = work.take_view()
  assert_true(!view.streaming_response())
  let handoff = view.handoff()
  let prompt = Bytes::makei(handoff.prompt_length(), i => {
    handoff.prompt_byte_at(i)
  })
  assert_true(@utf8.decode(prompt).contains("normalized image placeholders"))
  view.release()
  assert_true(workspace.normalized_body is None)
  let again = workspace.begin(
    inbound_http_view(
      2, "{\"model\":\"model-x\",\"messages\":[{\"role\":\"user\",\"content\":\"plain\"}]}",
    ),
    @inference.RequestId::new(10UL),
  )
  while again.state() is LunaOpenAIInboundPending {
    ignore(again.progress())
  }
  assert_eq(again.state(), LunaOpenAIInboundReady)
  again.take_view().release()
}

///|
test "stale source fails normalized admission and failed bodies release replacement storage" {
  let limits = openai_wb_fixture_limits()
  let workspace = inbound_workspace(limits, 1)
  let http = inbound_http_view(2, "source")
  let body = @utf8.encode(
    "{\"model\":\"model-x\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}",
  )
  let work = workspace.begin_normalized(
    http,
    @inference.RequestId::new(1UL),
    body,
  )
  http.release()
  ignore(work.progress())
  assert_eq(work.state(), LunaOpenAIInboundFailed)
  assert_true(workspace.normalized_body is None)
  work.abort()
  let invalid = workspace.begin_normalized(
    inbound_http_view(2, "other"),
    @inference.RequestId::new(2UL),
    b"{invalid}",
  )
  while invalid.state() is LunaOpenAIInboundPending {
    ignore(invalid.progress())
  }
  assert_eq(invalid.state(), LunaOpenAIInboundFailed)
  assert_true(workspace.normalized_body is None)
  invalid.abort()
}
