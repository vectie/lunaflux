///|
fn expect_fake_server_sse_rejection(event : String) -> Unit raise {
  let adapter = try! baseline_test_adapter()
  try! adapter.submit(0)
  try! adapter.observe_response_head(0, 200, "text/event-stream; charset=utf-8")
  try fake_server_offer(adapter, event) catch {
    OpenAIBaselineAmbiguousEvent
    | OpenAIBaselineInvalidSse
    | OpenAIBaselineInvalidUsage => ()
    _ => fail("unexpected fake-server rejection")
  } noraise {
    _ => fail("ambiguous fake-server event was accepted")
  }
}

///|
test "fake baseline server rejects ambiguous SSE semantics" {
  expect_fake_server_sse_rejection(
    "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":null},{\"index\":1,\"delta\":{},\"finish_reason\":null}]}\n\n",
  )
  expect_fake_server_sse_rejection(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[]},\"finish_reason\":null}]}\n\n",
  )
  expect_fake_server_sse_rejection(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"finish_reason\":\"content_filter\"}]}\n\n",
  )
  expect_fake_server_sse_rejection(
    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"prompt_tokens\":3,\"completion_tokens\":0,\"total_tokens\":3}}\n\n",
  )
  expect_fake_server_sse_rejection("event: message\ndata: [DONE]\n\n")
}

///|
test "DONE requires one finish and exact usage" {
  let adapter = try! baseline_test_adapter()
  try! adapter.submit(0)
  try! adapter.observe_response_head(0, 200, "text/event-stream")
  try fake_server_offer(adapter, "data: [DONE]\n\n") catch {
    OpenAIBaselineAmbiguousEvent => ()
    _ => fail("unexpected premature DONE rejection")
  } noraise {
    _ => fail("premature DONE was accepted")
  }
}

///|
test "content requires exact token observations even on transport failure" {
  expect_fake_server_sse_rejection(
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\n",
  )
  let adapter = try! baseline_test_adapter()
  try! adapter.submit(0)
  try! adapter.observe_response_head(0, 200, "text/event-stream")
  try! fake_server_offer(
    adapter, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"logprobs\":{\"content\":[{\"token\":\"x\"}]},\"finish_reason\":null}]}\n\n",
  )
  try! adapter.terminate_transport(0, Failed)
  let summary = try! adapter.finish().measurements().summary()
  assert_eq(summary.failed(), 1)
  assert_eq(summary.generated_tokens(), 1L)
}

///|
test "response head and transport outcome mappings are closed" {
  let redirect = try! baseline_test_adapter()
  try! redirect.submit(0)
  try redirect.observe_response_head(0, 302, "application/json") catch {
    OpenAIBaselineAmbiguousEvent => ()
    _ => fail("unexpected redirect rejection")
  } noraise {
    _ => fail("redirect was mapped to a benchmark outcome")
  }
  let wrong_media = try! baseline_test_adapter()
  try! wrong_media.submit(0)
  try wrong_media.observe_response_head(0, 200, "application/json") catch {
    OpenAIBaselineInvalidResponseHead => ()
    _ => fail("unexpected media-type rejection")
  } noraise {
    _ => fail("non-SSE success response was admitted")
  }
  let rejected = try! baseline_test_adapter()
  try! rejected.submit(0)
  try! rejected.observe_response_head(0, 429, "application/json")
  let rejected_summary = try! rejected.finish().measurements().summary()
  assert_eq(rejected_summary.rejected(), 1)
  let timed_out = try! baseline_test_adapter()
  try! timed_out.submit(0)
  try! timed_out.terminate_transport(0, TimedOut)
  assert_eq(try! timed_out.finish().measurements().summary().timed_out(), 1)
}

///|
test "bounded SSE storage rejects an oversized fake-server event" {
  let adapter = try! baseline_test_adapter(maximum_event_bytes=64)
  try! adapter.submit(0)
  try! adapter.observe_response_head(0, 200, "text/event-stream")
  try fake_server_offer(adapter, "data: " + "x".repeat(65)) catch {
    OpenAIBaselineEventTooLarge => ()
    _ => fail("unexpected oversized-event rejection")
  } noraise {
    _ => fail("oversized event was buffered")
  }
}
