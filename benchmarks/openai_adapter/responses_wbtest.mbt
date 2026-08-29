///|
fn responses_test_observer(
  engine? : @evidence.BenchmarkEngine = LunaFlux,
) -> OpenAIResponsesTrialObserver raise {
  OpenAIResponsesTrialContract::new(
    baseline_test_identity(engine~),
    baseline_test_inputs(),
    1,
    0,
    2048,
    16,
  ).start()
}

///|
fn responses_test_offer(
  observer : OpenAIResponsesTrialObserver,
  event : String,
) -> Unit raise {
  observer.offer_sse_bytes(0, @utf8.encode(event))
}

///|
fn responses_test_begin(observer : OpenAIResponsesTrialObserver) -> Unit raise {
  observer.submit(0)
  observer.observe_response_head(0, 200, "text/event-stream")
  responses_test_offer(
    observer, "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"tiny-bf16\",\"status\":\"in_progress\"}}\n\n",
  )
}

///|
fn responses_test_complete(
  observer : OpenAIResponsesTrialObserver,
) -> OpenAIResponsesTrialEvidence raise {
  responses_test_begin(observer)
  let split = @utf8.encode(
    "data: {\"type\":\"response.output_text.delta\",\"response_id\":\"resp_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"x\"}\n\n",
  )
  for byte in split {
    observer.offer_sse_bytes(0, Bytes::from_array([byte]))
  }
  responses_test_offer(
    observer, "data: {\"type\":\"response.output_text.delta\",\"response_id\":\"resp_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"y\"}\r\n\r\n",
  )
  responses_test_offer(
    observer, "data: {\"type\":\"response.usage\",\"response_id\":\"resp_1\",\"usage\":{\"input_tokens\":3,\"cached_input_tokens\":1,\"output_tokens\":2,\"total_tokens\":5}}\n\n",
  )
  responses_test_offer(
    observer, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"tiny-bf16\",\"status\":\"completed\",\"finish_reason\":\"stop\",\"output_text\":\"\",\"usage\":{\"input_tokens\":3,\"cached_input_tokens\":1,\"output_tokens\":2,\"total_tokens\":5}}}\n\n",
  )
  responses_test_offer(observer, "data: [DONE]\n\n")
  observer.finish()
}

///|
test "LunaFlux Responses lifecycle produces authority-free profile measurements" {
  let evidence = responses_test_complete(responses_test_observer())
  let summary = evidence.measurements().summary()
  assert_eq(evidence.engine_identity().engine(), LunaFlux)
  assert_eq(evidence.workload().profile(), Latency)
  assert_eq(summary.submitted(), 1)
  assert_eq(summary.completed(), 1)
  assert_eq(summary.input_tokens(), 3L)
  assert_eq(summary.generated_tokens(), 2L)
  assert_eq(evidence.observed_protocol(), ResponsesSseV1)
  assert_not_eq(evidence.protocol_digest(), baseline_test_digest('6'))
  let canonical = @utf8.decode(evidence.canonical_bytes())
  assert_true(
    canonical.contains("observation_protocol=openai.responses.sse.v1"),
  )
  assert_true(canonical.contains("comparison_authority=none"))
  assert_true(canonical.contains("correctness_authority=none"))
}

///|
test "Responses observer preserves all three pinned engine identities" {
  for engine in [@evidence.LunaFlux, Vllm, Sglang] {
    let evidence = responses_test_complete(responses_test_observer(engine~))
    assert_eq(evidence.engine_identity().engine(), engine)
  }
}

///|
test "Responses observer rejects reordered and inconsistent lifecycle events" {
  let reordered = responses_test_observer()
  reordered.submit(0)
  reordered.observe_response_head(0, 200, "text/event-stream; charset=utf-8")
  try
    responses_test_offer(
      reordered, "data: {\"type\":\"response.output_text.delta\",\"response_id\":\"resp_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"x\"}\n\n",
    )
  catch {
    OpenAIBaselineAmbiguousEvent => ()
    _ => fail("unexpected reordered Responses rejection")
  } noraise {
    _ => fail("delta before response.created was accepted")
  }

  let mismatched = responses_test_observer()
  responses_test_begin(mismatched)
  try
    responses_test_offer(
      mismatched, "data: {\"type\":\"response.output_text.delta\",\"response_id\":\"resp_2\",\"output_index\":0,\"content_index\":0,\"delta\":\"x\"}\n\n",
    )
  catch {
    OpenAIBaselineAmbiguousEvent => ()
    _ => fail("unexpected response-id mismatch rejection")
  } noraise {
    _ => fail("mismatched response id was accepted")
  }
}

///|
test "Responses failed event without output maps to exact failed outcome" {
  let observer = responses_test_observer()
  responses_test_begin(observer)
  responses_test_offer(
    observer, "data: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_1\",\"status\":\"failed\",\"error\":{\"code\":\"device-failed\",\"message\":\"request failed\",\"retryable\":false}}}\n\n",
  )
  responses_test_offer(observer, "data: [DONE]\n\n")
  let summary = observer.finish().measurements().summary()
  assert_eq(summary.failed(), 1)
  assert_eq(summary.generated_tokens(), 0L)
}

///|
test "Responses partial transport failure without usage fails closed" {
  let observer = responses_test_observer()
  responses_test_begin(observer)
  responses_test_offer(
    observer, "data: {\"type\":\"response.output_text.delta\",\"response_id\":\"resp_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"fragment\"}\n\n",
  )
  try observer.terminate_transport(0, Failed) catch {
    OpenAIBaselineAmbiguousEvent => ()
    _ => fail("unexpected partial Responses transport rejection")
  } noraise {
    _ => fail("partial Responses output invented a generated-token count")
  }
}

///|
test "shared Responses lifecycle accepts terminal-only usage and auxiliary events" {
  let observer = responses_test_observer(engine=Sglang)
  observer.submit(0)
  observer.observe_response_head(0, 200, "text/event-stream")
  responses_test_offer(
    observer, "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"in_progress\",\"model\":\"model\",\"max_output_tokens\":16}}\n\n",
  )
  responses_test_offer(
    observer, "data: {\"type\":\"response.in_progress\",\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\"}}\n\n",
  )
  responses_test_offer(
    observer, "data: {\"type\":\"response.output_item.added\",\"response_id\":\"resp_1\",\"output_index\":0,\"item\":{\"type\":\"message\"}}\n\n",
  )
  responses_test_offer(
    observer, "event: response.output_text.delta\r\ndata: {\"type\":\"response.output_text.delta\",\"response_id\":\"resp_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"hello\"}\r\n\r\n",
  )
  responses_test_offer(
    observer, "data: {\"type\":\"response.output_text.done\",\"response_id\":\"resp_1\",\"output_index\":0,\"content_index\":0,\"text\":\"hello\"}\n\n",
  )
  responses_test_offer(
    observer, "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"completed\",\"usage\":{\"input_tokens\":3,\"input_tokens_details\":{\"cached_tokens\":1},\"output_tokens\":1,\"total_tokens\":4}}}\n\n",
  )
  responses_test_offer(observer, "data: [DONE]\n\n")
  let evidence = observer.finish()
  let summary = evidence.measurements().summary()
  assert_eq(evidence.engine_identity().engine(), Sglang)
  assert_eq(summary.completed(), 1)
  assert_eq(summary.generated_tokens(), 1L)
}
