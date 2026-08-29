///|
fn inbound_expect_json_failure(route : Int, body : String) -> Unit raise {
  let inference = openai_wb_fixture_limits()
  let workspace = inbound_workspace(inference, 1)
  let work = workspace.begin(
    inbound_http_view(route, body),
    @inference.RequestId::new(55UL),
  )
  while work.state() == LunaOpenAIInboundPending {
    ignore(work.progress())
  }
  assert_eq(work.state(), LunaOpenAIInboundFailed)
  assert_eq(work.failure(), LunaOpenAICompatInvalid(LunaOpenAICompatJson))
  work.abort()
  openai_expect_rule(() => ignore(work.state()), LunaOpenAICompatEpoch)
}

///|
test "Luna OpenAI inbound rejects duplicate unknown missing and trailing fields" {
  let responses : FixedArray[String] = [
    "{\"model\":\"model-x\",\"model\":\"model-x\",\"input\":\"x\",\"stream\":true}",
    "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,\"extra\":1}", "{\"model\":\"model-x\",\"stream\":true}",
    "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true,}", "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":false}",
    "{\"model\":\"wrong\",\"input\":\"x\",\"stream\":true}", "{\"model\":\"model-x\",\"input\":\"\",\"stream\":true}",
  ]
  for body in responses {
    inbound_expect_json_failure(1, body)
  }
  let chats : FixedArray[String] = [
    "{\"model\":\"model-x\",\"stream\":true,\"messages\":[]}", "{\"model\":\"model-x\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"x\",}]}",
    "{\"model\":\"model-x\",\"stream\":true,\"messages\":[{\"role\":\"assistant\",\"content\":\"x\"}]}",
    "{\"model\":\"model-x\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"x\"},{\"role\":\"user\",\"content\":\"y\"}]}",
    "{\"model\":\"model-x\",\"stream\":true,\"messages\":[{\"content\":\"\",\"role\":\"user\"}]}",
  ]
  for body in chats {
    inbound_expect_json_failure(2, body)
  }
}

///|
test "Luna OpenAI inbound rejects malformed Unicode and UTF-8 classes" {
  for
    body in [
      "{\"model\":\"model-x\",\"input\":\"\\uD800\",\"stream\":true}", "{\"model\":\"model-x\",\"input\":\"\\uDC00\",\"stream\":true}",
      "{\"model\":\"model-x\",\"input\":\"\\uD800\\u0041\",\"stream\":true}", "{\"model\":\"model-x\",\"input\":\"\\q\",\"stream\":true}",
    ] {
    inbound_expect_json_failure(1, body)
  }
  let prefix = inbound_bytes("{\"model\":\"model-x\",\"input\":\"")
  let suffix = inbound_bytes("\",\"stream\":true}")
  let raw : FixedArray[Byte] = FixedArray::make(
    prefix.length() + 2 + suffix.length(),
    0,
  )
  for index in 0..<prefix.length() {
    raw[index] = prefix[index]
  }
  raw[prefix.length()] = 0xc0
  raw[prefix.length() + 1] = 0x80
  for index in 0..<suffix.length() {
    raw[prefix.length() + 2 + index] = suffix[index]
  }
  let inference = openai_wb_fixture_limits()
  let work = inbound_workspace(inference, 1).begin(
    inbound_http_raw_view(1, raw),
    @inference.RequestId::new(1UL),
  )
  while work.state() == LunaOpenAIInboundPending {
    ignore(work.progress())
  }
  assert_eq(work.failure(), LunaOpenAICompatInvalid(LunaOpenAICompatJson))
  work.abort()
}

///|
test "Luna OpenAI inbound authenticates HTTP through final frame and reuses exactly" {
  let inference = openai_wb_fixture_limits()
  let workspace = inbound_workspace(inference, 1)
  let http = inbound_http_view(
    1, "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true}",
  )
  let work = workspace.begin(http, @inference.RequestId::new(1UL))
  ignore(work.progress())
  http.release()
  ignore(work.progress())
  assert_eq(work.state(), LunaOpenAIInboundFailed)
  assert_eq(work.failure(), LunaOpenAICompatInvalid(LunaOpenAICompatHttp))
  work.abort()
  let http2 = inbound_http_view(
    1, "{\"model\":\"model-x\",\"input\":\"y\",\"stream\":true}",
  )
  let work2 = workspace.begin(http2, @inference.RequestId::new(2UL))
  while work2.state() == LunaOpenAIInboundPending {
    ignore(work2.progress())
  }
  assert_false(http2.is_live())
  let view = work2.take_view()
  let length = view.length()
  let destination : FixedArray[Byte] = FixedArray::make(length, 0)
  openai_expect_rule(
    () => {
      ignore(
        view.copy_chunk_to(
          destination,
          destination_offset=0,
          source_offset=0,
          length=0,
        ),
      )
    },
    LunaOpenAICompatDestination,
  )
  view.release()
  openai_expect_rule(() => ignore(view.length()), LunaOpenAICompatEpoch)
}

///|
test "Luna OpenAI inbound HTTP receipt splits do not change canonical frame" {
  let inference = openai_wb_fixture_limits()
  let body = "{\"input\" : \"split\", \"stream\" : true, \"model\" : \"model-x\"}"
  let mut baseline : FixedArray[Byte] = []
  for chunk in [1, 2, 17, 2147483647] {
    let (frame, _) = inbound_encode(
      inbound_workspace(inference, 17),
      inbound_http_view(1, body, chunk~),
    )
    if baseline.length() == 0 {
      baseline = frame
    } else {
      openai_wb_assert_same(frame, baseline)
    }
  }
}

///|
test "Luna OpenAI chat template bound is enforced before an oversized write" {
  let inference = openai_wb_fixture_limits()
  let body = "{\"model\":\"model-x\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"12345678901234567890\"}]}"
  let workspace = inbound_workspace(inference, 1, template_maximum=25)
  let work = workspace.begin(
    inbound_http_view(2, body),
    @inference.RequestId::new(9UL),
  )
  while work.state() == LunaOpenAIInboundPending {
    ignore(work.progress())
  }
  assert_eq(work.failure(), LunaOpenAICompatInvalid(LunaOpenAICompatCapacity))
  assert_eq(workspace.prompt_length, 25)
  work.abort()
}
