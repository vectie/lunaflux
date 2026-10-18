///|
test "stream false and omitted stream select buffered presentation without changing engine semantics" {
  let inference = openai_wb_fixture_limits()
  let workspace = inbound_workspace(inference, 1)
  for
    (field, streaming) in [
      ("", false),
      (",\"stream\":false", false),
      (",\"stream\":true", true),
      ("", false),
    ] {
    let work = workspace.begin(
      inbound_http_view(
        1,
        "{\"model\":\"model-x\",\"input\":\"x\"" + field + "}",
      ),
      @inference.RequestId::new(7UL),
    )
    while work.state() == LunaOpenAIInboundPending {
      ignore(work.progress())
    }
    assert_eq(work.state(), LunaOpenAIInboundReady)
    let view = work.take_view()
    assert_eq(view.streaming_response(), streaming)
    assert_eq(
      view.handoff().stream_preference(),
      @inference.StreamPreference::Streaming,
    )
    view.release()
  }
  for value in ["falsee", "False", "0", "\"false\""] {
    inbound_expect_json_failure(
      1,
      "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":" + value + "}",
    )
  }
}
