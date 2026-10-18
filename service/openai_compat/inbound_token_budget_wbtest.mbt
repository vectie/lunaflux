///|
fn request_budget(
  bytes : FixedArray[Byte],
  inference : @inference.InferenceLimits,
) -> Int raise {
  @framed_wire.RequestFrameBuffer::new(
    @framed_wire.FramedWireLimits::new(inference, max_frame_bytes=4096),
  )
  .load(bytes, bytes.length())
  .request()
  .max_new_tokens()
}

///|
test "request output budgets exceed defaults but not engine capacity across work budgets" {
  let inference = openai_wb_fixture_limits()
  for budget in [1, 17, 65536] {
    let workspace = inbound_workspace(inference, budget)
    for
      (route, suffix, value) in [
        (2, ",\"max_tokens\":9", 9),
        (2, ",\"max_completion_tokens\":12", 12),
        (2, ",\"max_tokens\":9,\"max_completion_tokens\":7", 7),
        (2, ",\"max_completion_tokens\":7,\"max_tokens\":9", 7),
        (1, ",\"max_output_tokens\":11", 11),
        (1, "", 4),
        (2, "", 4),
        (1, ",\"max_output_tokens\":1", 1),
      ] {
      let prompt = if route == 1 {
        "\"input\":\"x\""
      } else {
        "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]"
      }
      let body = "{\"model\":\"model-x\",\"stream\":true," +
        prompt +
        suffix +
        "}"
      let (frame, _) = inbound_encode(workspace, inbound_http_view(route, body))
      assert_eq(request_budget(frame, inference), value)
    }
  }
}

///|
test "malformed or out of range request budgets fail without publishing and allow reuse" {
  let inference = openai_wb_fixture_limits()
  let workspace = inbound_workspace(inference, 1)
  for
    suffix in [
      ",\"max_output_tokens\":0", ",\"max_output_tokens\":13", ",\"max_output_tokens\":-1",
      ",\"max_output_tokens\":1.5", ",\"max_output_tokens\":4294967297", ",\"max_output_tokens\":18446744073709551615",
      ",\"max_output_tokens\":2,\"max_output_tokens\":3", ",\"max_tokens\":2", ",\"max_completion_tokens\":2",
    ] {
    let work = workspace.begin(
      inbound_http_view(
        1,
        "{\"model\":\"model-x\",\"stream\":true,\"input\":\"x\"" + suffix + "}",
      ),
      @inference.RequestId::new(55UL),
    )
    while work.state() == LunaOpenAIInboundPending {
      ignore(work.progress())
    }
    assert_eq(work.state(), LunaOpenAIInboundFailed)
    assert_eq(work.failure(), LunaOpenAICompatInvalid(LunaOpenAICompatJson))
    assert_eq(workspace.handoff_writes.length(), 0)
    work.abort()
    let (frame, _) = inbound_encode(
      workspace,
      inbound_http_view(
        1, "{\"model\":\"model-x\",\"stream\":true,\"input\":\"x\"}",
      ),
    )
    assert_eq(request_budget(frame, inference), 4)
  }
}

///|
test "request output budget preserves prompt room below deployed context ceiling" {
  let inference = openai_wb_fixture_limits()
  let workspace = inbound_workspace(
    inference,
    17,
    max_new_tokens=4,
    context_ceiling=8,
  )
  let work = workspace.begin(
    inbound_http_view(
      1, "{\"model\":\"model-x\",\"stream\":true,\"input\":\"x\",\"max_output_tokens\":8}",
    ),
    @inference.RequestId::new(55UL),
  )
  while work.state() == LunaOpenAIInboundPending {
    ignore(work.progress())
  }
  assert_eq(work.state(), LunaOpenAIInboundFailed)
  work.abort()
  let (frame, _) = inbound_encode(
    workspace,
    inbound_http_view(
      1, "{\"model\":\"model-x\",\"stream\":true,\"input\":\"x\",\"max_output_tokens\":7}",
    ),
  )
  assert_eq(request_budget(frame, inference), 7)
}
