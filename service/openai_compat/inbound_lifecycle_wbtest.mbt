///|
test "Luna OpenAI inbound pins Busy abort stale reuse and epoch exhaustion" {
  let inference = openai_wb_fixture_limits()
  let workspace = inbound_workspace(inference, 1)
  let body = "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true}"
  let http = inbound_http_view(1, body)
  let work = workspace.begin(http, @inference.RequestId::new(1UL))
  let busy_http = inbound_http_view(1, body)
  let busy_id = @inference.RequestId::new(2UL)
  openai_expect_rule(
    () => ignore(workspace.begin(busy_http, busy_id)),
    LunaOpenAICompatLifecycle,
  )
  assert_true(busy_http.is_live())
  busy_http.release()
  work.abort()
  assert_false(http.is_live())
  openai_expect_rule(() => ignore(work.state()), LunaOpenAICompatEpoch)
  let work2 = workspace.begin(
    inbound_http_view(1, body),
    @inference.RequestId::new(3UL),
  )
  while work2.state() == LunaOpenAIInboundPending {
    ignore(work2.progress())
  }
  work2.abort()
  workspace.epoch = 0xffffffffffffffffUL
  let final_http = inbound_http_view(1, body)
  let final_id = @inference.RequestId::new(4UL)
  openai_expect_rule(
    () => ignore(workspace.begin(final_http, final_id)),
    LunaOpenAICompatEpoch,
  )
  assert_true(final_http.is_live())
  final_http.release()
}

///|
test "Luna OpenAI inbound stale HTTP fails every retained phase" {
  let inference = openai_wb_fixture_limits()
  let targets : FixedArray[Int] = [
    LUNA_INBOUND_TOP_KEY,
    LUNA_INBOUND_STRING,
    LUNA_INBOUND_STRING_EMIT,
    LUNA_INBOUND_PARSE_FINAL,
    LUNA_INBOUND_RENDER_STRING,
    LUNA_INBOUND_RENDER_EMIT,
    LUNA_INBOUND_HANDOFF_PUSH,
    LUNA_INBOUND_HANDOFF_PROGRESS,
    LUNA_INBOUND_FINAL,
  ]
  let body = "{\"model\":\"model-x\",\"input\":\"a\\nb\",\"stream\":true}"
  for target in targets {
    let workspace = inbound_workspace(inference, 1)
    let http = inbound_http_view(1, body)
    let work = workspace.begin(http, @inference.RequestId::new(7UL))
    let mut turns = 0
    while workspace.phase != target &&
          work.state() == LunaOpenAIInboundPending &&
          turns < 10000 {
      ignore(work.progress())
      turns += 1
    }
    assert_eq(workspace.phase, target)
    http.release()
    ignore(work.progress())
    assert_eq(work.state(), LunaOpenAIInboundFailed)
    assert_eq(work.failure(), LunaOpenAICompatInvalid(LunaOpenAICompatHttp))
    work.abort()
  }
}

///|
test "Luna OpenAI inbound exact message capacity and duplicate message keys" {
  let inference = openai_wb_fixture_limits()
  let below = "{\"model\":\"model-x\",\"stream\":true,\"messages\":[{\"role\":\"system\",\"content\":\"zero\"},{\"role\":\"user\",\"content\":\"one\"}]}"
  let (below_frame, _) = inbound_encode(
    inbound_workspace(inference, 1, max_messages=3),
    inbound_http_view(2, below),
  )
  assert_eq(inbound_text(below_frame, inference), "<S>zero</S><U>one</U><A>")
  let exact = "{\"model\":\"model-x\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"one\"},{\"role\":\"assistant\",\"content\":\"two\"},{\"role\":\"user\",\"content\":\"three\"}]}"
  let (frame, _) = inbound_encode(
    inbound_workspace(inference, 1, max_messages=3),
    inbound_http_view(2, exact),
  )
  assert_eq(
    inbound_text(frame, inference),
    "<U>one</U><A>two</A><U>three</U><A>",
  )
  inbound_expect_json_failure(
    2, "{\"model\":\"model-x\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"role\":\"user\",\"content\":\"x\"}]}",
  )
  inbound_expect_json_failure(
    2, "{\"model\":\"model-x\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"x\",\"content\":\"y\"}]}",
  )
  let over = "{\"model\":\"model-x\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"1\"},{\"role\":\"assistant\",\"content\":\"2\"},{\"role\":\"user\",\"content\":\"3\"}]}"
  let work = inbound_workspace(inference, 1, max_messages=2).begin(
    inbound_http_view(2, over),
    @inference.RequestId::new(5UL),
  )
  while work.state() == LunaOpenAIInboundPending {
    ignore(work.progress())
  }
  assert_eq(work.failure(), LunaOpenAICompatInvalid(LunaOpenAICompatJson))
  work.abort()
}

///|
test "Luna OpenAI inbound role sequence boundaries are strict" {
  for
    body in [
      "{\"model\":\"model-x\",\"stream\":true,\"messages\":[{\"role\":\"system\",\"content\":\"s\"}]}",
      "{\"model\":\"model-x\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"u\"},{\"role\":\"assistant\",\"content\":\"a\"}]}",
    ] {
    inbound_expect_json_failure(2, body)
  }
  let inference = openai_wb_fixture_limits()
  let (frame, _) = inbound_encode(
    inbound_workspace(inference, 1),
    inbound_http_view(
      2, "{\"model\":\"model-x\",\"stream\":true,\"messages\":[{\"role\":\"system\",\"content\":\"s\"},{\"role\":\"user\",\"content\":\"u\"}]}",
    ),
  )
  assert_eq(inbound_text(frame, inference), "<S>s</S><U>u</U><A>")
}

///|
test "Luna OpenAI inbound sizing is exact and forged limits fail before allocation" {
  let inference = openai_wb_fixture_limits()
  assert_eq(LunaOpenAIInboundWorkspace::required_byte_cells(inference), 448UL)
  assert_eq(
    LunaOpenAIInboundWorkspace::required_int_cells(
      inference,
      LunaOpenAIInboundLimits::new(max_messages=64),
    ),
    202UL,
  )
  assert_eq(LunaOpenAIInboundWorkspace::required_reference_cells(), 8UL)
  let forged : LunaOpenAIInboundLimits = { max_messages: 2147483647 }
  openai_expect_rule(
    () => {
      ignore(LunaOpenAIInboundWorkspace::required_int_cells(inference, forged))
    },
    LunaOpenAICompatLimits,
  )
  let model = inbound_bytes("model-x")
  let prefix = inbound_bytes("resp_")
  let forged_budget : LunaOpenAICompatStepBudget = { work_units: 0 }
  openai_expect_rule(
    () => {
      ignore(
        LunaOpenAICompatWorkspace::new(
          inference,
          forged_budget,
          model,
          model_alias_offset=0,
          model_alias_length=model.length(),
          prefix,
          response_id_prefix_offset=0,
          response_id_prefix_length=prefix.length(),
        ),
      )
    },
    LunaOpenAICompatWorkUnits,
  )
  let valid_template = inbound_template(inference)
  let invalid_template : LunaOpenAIChatTemplate = {
    storage: [],
    offsets: [],
    lengths: [],
    max_rendered_prompt_bytes: 1,
  }
  let identity = inbound_identity()
  let scope = @inference.CacheScope::from_ascii(b"openai", inference)
  let seed = @inference.SamplingSeed::new(1UL)
  let deadline = @inference.DeadlineBudget::from_milliseconds(10UL, inference)
  let valid_budget = LunaOpenAICompatStepBudget::new(1)
  let valid_limits = LunaOpenAIInboundLimits::new(max_messages=1)
  let make_inbound = fn(
    chosen_budget : LunaOpenAICompatStepBudget,
    chosen_limits : LunaOpenAIInboundLimits,
    chosen_template : LunaOpenAIChatTemplate,
  ) -> Unit raise LunaOpenAICompatError {
    ignore(
      LunaOpenAIInboundWorkspace::new(
        inference,
        chosen_budget,
        chosen_limits,
        chosen_template,
        model,
        model_alias_offset=0,
        model_alias_length=model.length(),
        identity,
        scope,
        max_new_tokens=1,
        context_ceiling=2,
        @inference.SamplingParameters::greedy(),
        seed,
        deadline,
      ),
    )
  }
  openai_expect_rule(
    () => make_inbound(forged_budget, valid_limits, valid_template),
    LunaOpenAICompatWorkUnits,
  )
  openai_expect_rule(
    () => make_inbound(valid_budget, forged, valid_template),
    LunaOpenAICompatLimits,
  )
  openai_expect_rule(
    () => make_inbound(valid_budget, valid_limits, invalid_template),
    LunaOpenAICompatTemplate,
  )
}

///|
test "Luna OpenAI inbound rejects zero text capacity before accepting HTTP" {
  let inference = openai_wb_fixture_limits()
  try
    inbound_workspace(inference, 1, max_new_tokens=4, context_ceiling=4)
  catch {
    LunaOpenAICompatInvalid(rule) => assert_eq(rule, LunaOpenAICompatLimits)
    _ => fail("unexpected zero-capacity inbound error")
  } noraise {
    _ => fail("zero-capacity inbound workspace was accepted")
  }
}
