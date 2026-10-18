///|
fn bytes(text : String) -> FixedArray[Byte] {
  let b = @utf8.encode(text)
  FixedArray::makei(b.length(), i => b[i])
}

///|
fn model_stops(
  root : @approved_fs.ApprovedRoot,
  vocabulary : Int,
) -> Array[Int] raise {
  let j = @json.parse(
    @utf8.decode(read_bounded(root, "generation_config.json", 1048576L)),
  )
  guard j is Object(fields) && fields.get("eos_token_id") is Some(ids) else {
    fail("missing model EOS declaration")
  }
  let ids = match ids {
    Array(values) => values
    value => [value]
  }
  let stops : Array[Int] = []
  for value in ids {
    guard value is Number(n, ..) &&
      n >= 0.0 &&
      n < vocabulary.to_double() &&
      n.to_int().to_double() == n &&
      !stops.contains(n.to_int()) else {
      fail("invalid model EOS declaration")
    }
    stops.push(n.to_int())
  }
  guard !stops.is_empty() && stops.length() <= 32 else {
    fail("model EOS count")
  }
  stops
}

///|
fn glm_template(
  inference : @inference.InferenceLimits,
) -> @openai.LunaOpenAIChatTemplate raise {
  @openai.LunaOpenAIChatTemplate::new(
    inference,
    bytes("<|system|>"),
    [],
    bytes("<|user|>"),
    [],
    bytes("<|assistant|>"),
    [],
    bytes("<|assistant|><think>"),
    document_prefix=bytes("[gMASK]<sop><|system|>Reasoning Effort: Low"),
    max_rendered_prompt_bytes=inference.max_text_bytes(),
  )
}

///|
async fn serve(
  plan : ServicePlan,
  manifest : String,
  root : @approved_fs.ApprovedRoot,
  kernels : @approved_fs.ApprovedRoot,
) -> Unit {
  let spec = inspect_spec(root)
  let inference = plan.inference(spec)
  let stops = model_stops(root, spec.vocabulary_size())
  let credential = read_bounded(kernels, plan.credential_file, 4096L)
  let secret = FixedArray::makei(credential.length(), i => credential[i])
  let auth = @auth.LunaApiAuthPolicy::new(
    secret,
    source_offset=0,
    length=secret.length(),
    maximum_credential_bytes=4096,
  )
  let tcp = @tcp.LunaOnlineTcpLimits::new(
    read_chunk_bytes=65536,
    write_chunk_bytes=65536,
    accept_timeout_millis=1,
    input_idle_timeout_millis=60000,
    write_timeout_millis=30000,
    reactor_transition_budget=64,
  )
  let prepared = prepare_service(plan, manifest, root, kernels, spec) catch {
    @session.LunaOnlineInstancePreparationFailed(cause) => {
      auth.close()
      fail("GLM preparation: \{Repr(cause)}")
    }
    error => {
      auth.close()
      raise error
    }
  }
  if prepared.state() != LunaOnlineInstanceReady {
    let cleanup = prepared.take_cleanup()
    let cause = cleanup.cause()
    while !cleanup.is_closed() {
      cleanup.retry_close()
    }
    auth.close()
    fail("GLM worker startup failed: \{Repr(cause)}; prepared resources closed")
  }
  let model = bytes("glm-5.3")
  let prefix = bytes("lf_")
  let pool = @tcp.bind_luna_online_openai_connection_pool(
    @socket.Addr::parse(plan.listen_address),
    @tcp.prepare_luna_online_openai_connection_pool(
      prepared,
      tcp,
      @tcp.LunaOnlineOpenAIConnectionPoolLimits::new(
        connection_capacity=plan.slots * 2 + 2,
        output_buffer_bytes=65536,
        io_poll_timeout_millis=1,
      ),
      @http.LunaHttp1Limits::new(max_head_bytes=16384, max_body_bytes=1048576),
      @http.LunaHttp1StepBudget::new(4096),
      @http.LunaHttp1ResponseStepBudget::new(4096),
      auth,
      @openai.LunaOpenAICompatStepBudget::new(4096),
      @openai.LunaOpenAICompatStepBudget::new(4096),
      @openai.LunaOpenAIInboundLimits::new(
        max_messages=64,
        required_stop_tokens=stops,
        template_response_input=true,
      ),
      glm_template(inference),
      model,
      model_alias_offset=0,
      model_alias_length=model.length(),
      prefix,
      response_id_prefix_offset=0,
      response_id_prefix_length=prefix.length(),
      @inference.CacheScope::from_ascii(b"glm", inference),
      max_new_tokens=512,
      context_ceiling=plan.context,
      @inference.SamplingParameters::greedy(),
      @inference.SamplingSeed::new(17UL),
      @inference.DeadlineBudget::from_milliseconds(3600000UL, inference),
      production_ready=plan.production_ready,
    ),
    tcp,
    log_capacity=256,
  )
  defer @async.protect_from_cancel(() => {
    pool.begin_drain()
    while pool.state() != LunaOnlineOpenAIConnectionPoolClosed {
      ignore(pool.progress_on_reactor())
      @async.pause()
    }
  })
  println(
    "GLM canonical HTTP worker ready at \{plan.listen_address}; production_ready=\{plan.production_ready}",
  )
  while pool.state() != LunaOnlineOpenAIConnectionPoolClosed {
    ignore(pool.progress_on_reactor())
    @async.pause()
  }
}
