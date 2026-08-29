///|
fn inbound_bytes(text : String) -> FixedArray[Byte] {
  let encoded = @utf8.encode(text)
  FixedArray::makei(encoded.length(), index => encoded[index])
}

///|
fn inbound_identity() -> @spec.ModelIdentity raise {
  @spec.ModelIdentity::new(
    @spec.ContentDigest::from_sha256(
      "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
    @spec.PlanDigest::from_sha256(
      "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    ),
  )
}

///|
fn inbound_template(
  inference : @inference.InferenceLimits,
  maximum? : Int = 64,
) -> LunaOpenAIChatTemplate raise {
  LunaOpenAIChatTemplate::new(
    inference,
    inbound_bytes("<S>"),
    inbound_bytes("</S>"),
    inbound_bytes("<U>"),
    inbound_bytes("</U>"),
    inbound_bytes("<A>"),
    inbound_bytes("</A>"),
    inbound_bytes("<A>"),
    max_rendered_prompt_bytes=maximum,
  )
}

///|
fn inbound_workspace(
  inference : @inference.InferenceLimits,
  budget : Int,
  template_maximum? : Int = 64,
  max_messages? : Int = 8,
  max_new_tokens? : Int = 4,
  context_ceiling? : Int = 16,
) -> LunaOpenAIInboundWorkspace raise {
  let model = inbound_bytes("model-x")
  LunaOpenAIInboundWorkspace::new(
    inference,
    LunaOpenAICompatStepBudget::new(budget),
    LunaOpenAIInboundLimits::new(max_messages~),
    inbound_template(inference, maximum=template_maximum),
    model,
    model_alias_offset=0,
    model_alias_length=model.length(),
    inbound_identity(),
    @inference.CacheScope::from_ascii(b"openai", inference),
    max_new_tokens~,
    context_ceiling~,
    @inference.SamplingParameters::greedy(),
    @inference.SamplingSeed::new(7UL),
    @inference.DeadlineBudget::from_milliseconds(5000UL, inference),
  )
}

///|
fn inbound_http_raw_view(
  route : Int,
  body_bytes : FixedArray[Byte],
  chunk? : Int = 2147483647,
) -> @http1.LunaHttp1View raise {
  let secret = inbound_bytes("secret")
  let policy = @api_auth.LunaApiAuthPolicy::new(
    secret,
    source_offset=0,
    length=secret.length(),
    maximum_credential_bytes=8,
  )
  let workspace = @http1.LunaHttp1Workspace::new(
    @http1.LunaHttp1Limits::new(max_head_bytes=1024, max_body_bytes=2048),
    @http1.LunaHttp1StepBudget::new(17),
    policy,
  )
  let path = if route == 1 { "/v1/responses" } else { "/v1/chat/completions" }
  let head = inbound_bytes(
    "POST " +
    path +
    " HTTP/1.1\r\n" +
    "Authorization: Bearer secret\r\n" +
    "Content-Length: " +
    body_bytes.length().to_string() +
    "\r\n" +
    "Content-Type: application/json\r\n\r\n",
  )
  let request : FixedArray[Byte] = FixedArray::make(
    head.length() + body_bytes.length(),
    0,
  )
  for index in 0..<head.length() {
    request[index] = head[index]
  }
  for index in 0..<body_bytes.length() {
    request[head.length() + index] = body_bytes[index]
  }
  let work = workspace.begin()
  let mut offset = 0
  while offset < request.length() {
    let consumed = work.offer(
      request,
      source_offset=offset,
      length=Int::min(chunk, request.length() - offset),
    )
    offset += consumed
    if consumed == 0 {
      ignore(work.progress())
    }
  }
  while work.state() != LunaHttp1Ready {
    ignore(work.progress())
  }
  work.take_view()
}

///|
fn inbound_http_view(
  route : Int,
  body : String,
  chunk? : Int = 2147483647,
) -> @http1.LunaHttp1View raise {
  inbound_http_raw_view(route, inbound_bytes(body), chunk~)
}

///|
fn inbound_encode(
  workspace : LunaOpenAIInboundWorkspace,
  http : @http1.LunaHttp1View,
  request_id? : UInt64 = 42UL,
) -> (FixedArray[Byte], UInt64) raise {
  let work = workspace.begin(http, @inference.RequestId::new(request_id))
  while work.state() == LunaOpenAIInboundPending {
    assert_eq(work.progress(), LunaOpenAIInboundAdvanced)
    assert_true(work.last_work_units() > 0)
    assert_true(work.last_work_units() <= workspace.budget.work_units())
  }
  if work.state() == LunaOpenAIInboundFailed {
    raise work.failure()
  }
  let total = work.total_work_units()
  let view = work.take_view()
  let handoff = view.handoff()
  let inference = handoff.inference_limits()
  let prompt = Bytes::makei(handoff.prompt_length(), index => {
    handoff.prompt_byte_at(index)
  })
  let semantic = handoff.semantic_view()
  let stop_tokens = Array::makei(semantic.stop_token_count(), index => {
    semantic.stop_token_at(index)
  })
  let stop_strings = Array::makei(semantic.stop_string_count(), string_index => {
    Bytes::makei(semantic.stop_string_length(string_index), byte_index => {
      semantic.stop_string_byte_at(string_index~, byte_index~)
    })
  })
  let cache_scope = Bytes::makei(semantic.cache_scope_length(), index => {
    semantic.cache_scope_byte_at(index)
  })
  let effective = handoff.effective_limits()
  let request = @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=handoff.request_id(),
    model_identity=handoff.model_identity(),
    input=@inference.Input::from_utf8(prompt[:], inference),
    max_new_tokens=effective.max_new_tokens(),
    context_ceiling=effective.context_ceiling(),
    sampling=handoff.validated_sampling().materialize(),
    sampling_seed=handoff.sampling_seed(),
    stops=@inference.StopConditions::new(
      stop_tokens[:],
      stop_strings[:],
      inference,
    ),
    stream_preference=handoff.stream_preference(),
    deadline=handoff.deadline(),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(cache_scope[:], inference),
      semantic.cache_permission(),
    ),
    trace=None,
    limits=inference,
  )
  let frame = @framed_wire.RequestFrameBuffer::new(
    @framed_wire.FramedWireLimits::new(inference, max_frame_bytes=4096),
  ).encode(request)
  let bytes : FixedArray[Byte] = FixedArray::make(frame.length(), 0)
  frame.copy_to(bytes, destination_offset=0)
  view.release()
  (bytes, total)
}

///|
fn inbound_text(
  frame : FixedArray[Byte],
  inference : @inference.InferenceLimits,
) -> String raise {
  let request = @framed_wire.RequestFrameBuffer::new(
      @framed_wire.FramedWireLimits::new(inference, max_frame_bytes=4096),
    )
    .load(frame, frame.length())
    .request()
  match request.input() {
    Text(text) => text.text()
    _ => abort("OpenAI inbound produced non-text input")
  }
}
