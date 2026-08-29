///|
fn baseline_test_digest(unit : Char) -> @evidence.BenchmarkDigest raise {
  @evidence.BenchmarkDigest::from_sha256(unit.to_string().repeat(64))
}

///|
fn baseline_test_identity(
  engine? : @evidence.BenchmarkEngine = Vllm,
) -> OpenAIBaselineIdentity raise {
  OpenAIBaselineIdentity::new(
    engine,
    baseline_test_digest('1'),
    baseline_test_digest('2'),
    baseline_test_digest('3'),
  )
}

///|
fn baseline_test_inputs(
  profile? : @evidence.BenchmarkProfile = Latency,
  tokenizer_unit? : Char = '4',
  corpus_unit? : Char = '5',
  protocol_unit? : Char = '6',
  workload_unit? : Char = '7',
  input_tokens? : FixedArray[Int64] = [3L],
) -> OpenAIBaselineInputs raise {
  OpenAIBaselineInputs::new(
    profile,
    baseline_test_digest(tokenizer_unit),
    baseline_test_digest(corpus_unit),
    baseline_test_digest(protocol_unit),
    baseline_test_digest(workload_unit),
    input_tokens,
  )
}

///|
fn baseline_test_adapter(
  engine? : @evidence.BenchmarkEngine = Vllm,
  maximum_event_bytes? : Int = 2048,
) -> OpenAIBaselineTrialAdapter raise {
  OpenAIBaselineTrialContract::new(
    baseline_test_identity(engine~),
    baseline_test_inputs(),
    1,
    0,
    maximum_event_bytes,
    16,
  ).start()
}

///|
fn fake_server_offer(
  adapter : OpenAIBaselineTrialAdapter,
  event : String,
) -> Unit raise {
  adapter.offer_sse_bytes(0, @utf8.encode(event))
}

///|
fn fake_server_complete_trial(
  engine? : @evidence.BenchmarkEngine = Vllm,
) -> OpenAIBaselineTrialEvidence raise {
  let adapter = baseline_test_adapter(engine~)
  drive_fake_server_completion(adapter)
}

///|
fn drive_fake_server_completion(
  adapter : OpenAIBaselineTrialAdapter,
) -> OpenAIBaselineTrialEvidence raise {
  adapter.submit(0)
  adapter.observe_response_head(0, 200, "text/event-stream")
  fake_server_offer(
    adapter, "data: {\"id\":\"r\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
  )
  let split = @utf8.encode(
    "data: {\"id\":\"r\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"logprobs\":{\"content\":[{\"token\":\"x\"}]},\"finish_reason\":null}]}\n\n",
  )
  for byte in split {
    adapter.offer_sse_bytes(0, Bytes::from_array([byte]))
  }
  fake_server_offer(
    adapter, "data: {\"id\":\"r\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"y\"},\"logprobs\":{\"content\":[{\"token\":\"y\"}]},\"finish_reason\":null}]}\n\n",
  )
  fake_server_offer(
    adapter, "data: {\"id\":\"r\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\r\n\r\n",
  )
  fake_server_offer(adapter, "data: [DONE]\n\n")
  adapter.finish()
}

///|
test "all three engines share the exact split Chat SSE lifecycle" {
  for engine in [@evidence.LunaFlux, Vllm, Sglang] {
    let evidence = fake_server_complete_trial(engine~)
    let summary = evidence.measurements().summary()
    assert_eq(evidence.engine_identity().engine(), engine)
    assert_eq(evidence.workload().profile(), Latency)
    assert_eq(summary.submitted(), 1)
    assert_eq(summary.completed(), 1)
    assert_eq(summary.input_tokens(), 3L)
    assert_eq(summary.generated_tokens(), 2L)
    assert_eq(evidence.tokenizer_digest(), baseline_test_digest('4'))
    assert_eq(evidence.corpus_digest(), baseline_test_digest('5'))
    assert_eq(evidence.observed_protocol(), ChatCompletionsSseV1)
    assert_eq(
      evidence.declared_request_protocol_digest(),
      baseline_test_digest('6'),
    )
    assert_not_eq(evidence.protocol_digest(), baseline_test_digest('6'))
    assert_not_eq(
      evidence.comparison_executable_digest(),
      baseline_test_digest('2'),
    )
    assert_true(
      @utf8.decode(evidence.canonical_bytes()).contains(
        "process_lifecycle_authority=external",
      ),
    )
  }
}

///|
test "caller digest cannot relabel Chat SSE evidence as Responses" {
  let evidence = fake_server_complete_trial()
  assert_eq(evidence.observed_protocol(), ChatCompletionsSseV1)
  assert_true(
    @utf8.decode(evidence.canonical_bytes()).contains(
      "observation_protocol=openai.chat.completions.sse.v1",
    ),
  )
  assert_true(
    @utf8.decode(evidence.canonical_bytes()).contains(
      "declared_request_protocol_sha256=" + baseline_test_digest('6').as_hex(),
    ),
  )
}

///|
test "comparison container identity binds image and engine configuration" {
  let canonical = baseline_test_identity()
  let changed = OpenAIBaselineIdentity::new(
    Vllm,
    baseline_test_digest('1'),
    baseline_test_digest('2'),
    baseline_test_digest('9'),
  )
  assert_not_eq(
    openai_baseline_comparison_executable(canonical),
    openai_baseline_comparison_executable(changed),
  )
}

///|
test "vLLM and SGLang declarations require identical shared inputs" {
  let canonical = baseline_test_inputs()
  require_identical_openai_comparison_inputs(
    canonical,
    baseline_test_inputs(),
    baseline_test_inputs(),
  )
  for
    hostile in [
      baseline_test_inputs(profile=Chat),
      baseline_test_inputs(tokenizer_unit='8'),
      baseline_test_inputs(corpus_unit='8'),
      baseline_test_inputs(protocol_unit='8'),
      baseline_test_inputs(workload_unit='8'),
      baseline_test_inputs(input_tokens=[4L]),
    ] {
    try require_identical_openai_baseline_inputs(canonical, hostile) catch {
      OpenAIBaselineInputMismatch => ()
      _ => fail("unexpected shared-input rejection")
    } noraise {
      _ => fail("mismatched baseline input declaration was accepted")
    }
  }
}
