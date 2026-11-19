///|
fn fixture(
  timeout : Int32,
  generate : async (
    @inference.LunaTextRequestHandoffLease,
    @event.LunaEventOwner,
    async (@event.LunaEventView) -> Unit,
  ) -> Unit,
) -> (@session_chat.Service, @auth.LunaApiAuthPolicy) raise {
  let limits = @inference.InferenceLimits::new(
    max_text_bytes=1024,
    max_input_tokens=64,
    max_new_tokens=8,
    max_context_tokens=64,
    max_token_id=255,
    max_stop_token_ids=4,
    max_stop_strings=2,
    max_stop_string_bytes=16,
    max_trace_bytes=16,
    max_cache_scope_bytes=16,
    max_decoded_delta_bytes=128,
    max_deadline_millis=5000UL,
    max_top_k=256,
    max_temperature=2.0,
  )
  let model = @utf8.encode("model-x").to_fixedarray()
  let prefix = @utf8.encode("resp_").to_fixedarray()
  let key = @utf8.encode("secret").to_fixedarray()
  let policy = @auth.LunaApiAuthPolicy::new(
    key,
    source_offset=0,
    length=key.length(),
    maximum_credential_bytes=64,
  )
  let identity = @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256("0".repeat(64)),
    @spec.PlanDigest::from_sha256("0".repeat(64)),
  )
  fn bytes(text : String) -> FixedArray[Byte] {
    @utf8.encode(text).to_fixedarray()
  }
  let template = @compat.LunaOpenAIChatTemplate::new(
    limits,
    bytes("<S>"),
    bytes("</S>"),
    bytes("<U>"),
    bytes("</U>"),
    bytes("<A>"),
    bytes("</A>"),
    bytes("<A>"),
    max_rendered_prompt_bytes=1024,
  )
  let inbound = @compat.LunaOpenAIInboundWorkspace::new(
    limits,
    @compat.LunaOpenAICompatStepBudget::new(128),
    @compat.LunaOpenAIInboundLimits::new(max_messages=2, required_stop_tokens=[
      255,
    ]),
    template,
    model,
    model_alias_offset=0,
    model_alias_length=model.length(),
    identity,
    @inference.CacheScope::from_ascii(b"test", limits),
    max_new_tokens=4,
    context_ceiling=64,
    @inference.SamplingParameters::greedy(),
    @inference.SamplingSeed::new(1UL),
    @inference.DeadlineBudget::from_milliseconds(timeout.to_uint64(), limits),
  )
  let presentation = @compat.LunaOpenAICompatWorkspace::new(
    limits,
    @compat.LunaOpenAICompatStepBudget::new(128),
    model,
    model_alias_offset=0,
    model_alias_length=model.length(),
    prefix,
    response_id_prefix_offset=0,
    response_id_prefix_length=prefix.length(),
    buffered=true,
  )
  (
    @session_chat.Service::new(
      limits,
      policy,
      inbound,
      presentation,
      timeout_ms=timeout,
      generate~,
      ready=() => true,
    ),
    policy,
  )
}

///|
fn request(streaming~ : Bool = false) -> Json {
  {
    "model": "model-x",
    "stream": streaming,
    "temperature": 0,
    "max_tokens": 4,
    "messages": [
      { "role": "system", "content": "policy" },
      { "role": "user", "content": "task" },
    ],
  }
}

///|
async fn post(
  url : String,
  document : Json,
  authorized~ : Bool = true,
) -> (@http.Response, Json) {
  let bytes = @utf8.encode(document.stringify())
  let headers : @http.Headers = {
    "Content-Type": "application/json",
    "Content-Length": bytes.length().to_string(),
  }
  if authorized {
    headers["Authorization"] = "Bearer secret"
  }
  let client = @http.Client(
    url.strip_suffix("/v1/chat/completions").unwrap().to_owned(),
  )
  defer client.close()
  let response = client.post(
    "/v1/chat/completions",
    bytes,
    extra_headers=headers,
  )
  (response, client.read_all().json())
}

///|
async fn publish(
  events : @event.LunaEventOwner,
  emit : async (@event.LunaEventView) -> Unit,
) -> Unit {
  let view = events.view()
  defer events.retire(view)
  emit(view)
}

///|
async fn reply(
  handoff : @inference.LunaTextRequestHandoffLease,
  events : @event.LunaEventOwner,
  emit : async (@event.LunaEventView) -> Unit,
  reason~ : @inference.FinishReason = StopSequence,
) -> Unit {
  let prompt = Bytes::makei(handoff.prompt_length(), index => {
    handoff.prompt_byte_at(index)
  })
  assert_eq(@utf8.decode(prompt), "<S>policy</S><U>task</U><A>")
  events.publish_accepted(handoff.model_identity(), handoff.effective_limits())
  publish(events, emit)
  let text = @utf8.encode("{\"skill_id\":\"one\",\"parameters\":{}}").to_fixedarray()
  assert_true(
    events.publish_token_status(
      token_id=7,
      position=0,
      text,
      source_offset=0,
      length=text.length(),
    )
    is LunaEventPublishReady,
  )
  publish(events, emit)
  events.publish_completed(
    reason,
    input_tokens=5,
    cached_input_tokens=0,
    output_tokens=2,
    [],
    tail_offset=0,
    tail_length=0,
  )
  publish(events, emit)
}

///|
async test "buffered canonical chat crosses real HTTP and reuses parser and renderer" {
  @async.with_timeout(5000, () => {
    @async.with_task_group() <| group => {
      let calls = Ref(0)
      let (service, policy) = fixture(2000, (handoff, events, emit) => {
        calls.val += 1
        if calls.val == 3 {
          events.publish_accepted(
            handoff.model_identity(),
            handoff.effective_limits(),
          )
          publish(events, emit)
          let code = @utf8.encode("backend_failure").to_fixedarray()
          events.publish_failed(
            code,
            code_offset=0,
            code_length=code.length(),
            retryable=true,
          )
          publish(events, emit)
        } else {
          reply(
            handoff,
            events,
            emit,
            reason=if calls.val == 1 { StopSequence } else { TokenLimit },
          )
        }
      })
      defer policy.close()
      defer service.close()
      let listener = @socket.TcpServer(@socket.Addr::parse("127.0.0.1:0"))
      defer listener.close()
      group.spawn_bg(no_wait=true, () => service.serve(listener))
      let url = "http://127.0.0.1:\{listener.addr.port()}/v1/chat/completions"
      for index in 0..<2 {
        let (response, data) = post(url, request())
        assert_eq(response.code, 200)
        let expected = if index == 0 { "stop" } else { "length" }
        assert_true(
          data
          is {
            "object": "chat.completion",
            "model": "model-x",
            "choices": [
              {
                "finish_reason": String(reason),
                "message": {
                  "role": "assistant",
                  "content": String(content),
                  ..
                },
                ..
              },
            ],
            "usage": {
              "prompt_tokens": 5,
              "completion_tokens": 2,
              "total_tokens": 7,
              ..
            },
            ..
          } &&
          reason == expected &&
          content == "{\"skill_id\":\"one\",\"parameters\":{}}",
        )
        while !service.is_ready() {
          @async.pause()
        }
      }
      let (unauthorized, _) = post(url, request(), authorized=false)
      assert_eq(unauthorized.code, 401)
      let malformed : Json = {
        "model": "model-x",
        "stream": true,
        "messages": [{ "role": "assistant", "content": "task" }],
      }
      let (unsupported, _) = post(url, malformed)
      assert_eq(unsupported.code, 400)
      assert_eq(calls.val, 2)
      assert_true(service.is_ready())
      let (failed, document) = post(url, request())
      assert_eq(failed.code, 500)
      assert_true(
        document
        is { "error": { "code": "backend_failure", "retryable": True, .. }, .. },
      )
      while !service.is_ready() {
        @async.pause()
      }
    }
  })
}

///|
async test "capacity timeout and peer disconnect retire callback before reuse" {
  @async.with_timeout(5000, () => {
    @async.with_task_group() <| group => {
      let entered = Ref(false)
      let retired = Ref(0)
      let delayed = Ref(true)
      let (service, policy) = fixture(150, (handoff, events, emit) => {
        entered.val = true
        defer {
          retired.val += 1
        }
        if delayed.val {
          @async.sleep(1000)
        }
        reply(handoff, events, emit)
      })
      defer policy.close()
      defer service.close()
      let listener = @socket.TcpServer(@socket.Addr::parse("127.0.0.1:0"))
      defer listener.close()
      group.spawn_bg(no_wait=true, () => service.serve(listener))
      let url = "http://127.0.0.1:\{listener.addr.port()}/v1/chat/completions"
      let first = group.spawn(() => post(url, request()).0.code)
      while !entered.val {
        @async.pause()
      }
      let (busy, _) = post(url, request())
      assert_eq(busy.code, 429)
      assert_true(first.wait() != 200)
      assert_eq(retired.val, 1)
      assert_true(service.is_ready())
      entered.val = false
      let peer = @socket.Tcp::connect(
        @socket.Addr::parse("127.0.0.1:\{listener.addr.port()}"),
      )
      let body = @utf8.encode(request().stringify())
      peer.write(
        @utf8.encode(
          "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\nContent-Length: \{body.length()}\r\n\r\n",
        ),
      )
      peer.write(body)
      while !entered.val {
        @async.pause()
      }
      peer.close()
      while retired.val != 2 || !service.is_ready() {
        @async.pause()
      }
      delayed.val = false
      let (reused, _) = post(url, request())
      assert_eq(reused.code, 200)
      assert_eq(retired.val, 3)
    }
  })
}
