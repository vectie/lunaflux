///|
fn prepare_openai_pool_e2e(
  executable : Bytes,
  tcp_limits : @online_tcp.LunaOnlineTcpLimits,
  fixture_root? : String = "/private/tmp",
) -> @online_tcp.LunaOnlineOpenAIConnectionPoolPreparation raise {
  let inference = tcp_inference_limits()
  let model = openai_e2e_bytes("model-x")
  let prefix = openai_e2e_bytes("resp_")
  let secret = openai_e2e_bytes("secret")
  let policy = @api_auth.LunaApiAuthPolicy::new(
    secret,
    source_offset=0,
    length=secret.length(),
    maximum_credential_bytes=8,
  )
  @online_tcp.prepare_luna_online_openai_connection_pool(
    prepare_tcp_service(executable, session_tokenizer(), fixture_root~),
    tcp_limits,
    @online_tcp.LunaOnlineOpenAIConnectionPoolLimits::new(
      connection_capacity=2,
      output_buffer_bytes=16384,
      io_poll_timeout_millis=1,
    ),
    @http1.LunaHttp1Limits::new(max_head_bytes=1024, max_body_bytes=512),
    @http1.LunaHttp1StepBudget::new(1),
    @http1.LunaHttp1ResponseStepBudget::new(17),
    policy,
    @openai_compat.LunaOpenAICompatStepBudget::new(1),
    @openai_compat.LunaOpenAICompatStepBudget::new(17),
    @openai_compat.LunaOpenAIInboundLimits::new(max_messages=2),
    openai_e2e_template(inference),
    model,
    model_alias_offset=0,
    model_alias_length=model.length(),
    prefix,
    response_id_prefix_offset=0,
    response_id_prefix_length=prefix.length(),
    @inference.CacheScope::from_ascii(b"scope", inference),
    max_new_tokens=1,
    context_ceiling=4,
    @inference.SamplingParameters::greedy(),
    @inference.SamplingSeed::new(17UL),
    @inference.DeadlineBudget::from_milliseconds(60000UL, inference),
  )
}

///|
async fn progress_openai_pool_to_connections(
  pool : @online_tcp.LunaOnlineOpenAIConnectionPool,
  expected : Int,
  stage : String,
) -> Unit {
  let mut turns = 0
  let mut accepted = 0
  let mut retired = 0
  let mut advanced = 0
  let mut backpressured = 0
  while pool.active_connection_count() != expected {
    match pool.progress_on_reactor() {
      LunaOnlineOpenAIConnectionAccepted => accepted += 1
      LunaOnlineOpenAIConnectionRetired => retired += 1
      LunaOnlineOpenAIConnectionPoolAdvanced => advanced += 1
      LunaOnlineOpenAIConnectionPoolBackpressured => backpressured += 1
      _ => ()
    }
    turns += 1
    if turns > 8192 {
      raise Failure(
        "OpenAI pool connection count did not converge stage=\{stage} active=\{pool.active_connection_count()} accepted=\{accepted} retired=\{retired} advanced=\{advanced} backpressured=\{backpressured}",
      )
    }
    if turns % 64 == 0 {
      @async.pause()
    }
  }
}

///|
async fn drain_openai_pool(
  pool : @online_tcp.LunaOnlineOpenAIConnectionPool,
) -> Unit {
  pool.begin_drain()
  let mut turns = 0
  while pool.state() != LunaOnlineOpenAIConnectionPoolClosed {
    ignore(pool.progress_on_reactor())
    turns += 1
    if turns > 1000000 {
      raise Failure("OpenAI pool drain did not converge")
    }
    if turns % 64 == 0 {
      @async.pause()
    }
  }
}

///|
async fn run_luna_openai_pool_evidence(
  executable : Bytes,
  fixture_root? : String = "/private/tmp",
) -> Unit {
  println("openai-pool-stage=bind")
  let tcp_limits = @online_tcp.LunaOnlineTcpLimits::new(
    read_chunk_bytes=2048,
    write_chunk_bytes=16384,
    accept_timeout_millis=25,
    input_idle_timeout_millis=60000,
    write_timeout_millis=60000,
    reactor_transition_budget=1,
  )
  let pool = @online_tcp.bind_luna_online_openai_connection_pool(
    @socket.Addr::parse("127.0.0.1:0"),
    prepare_openai_pool_e2e(executable, tcp_limits, fixture_root~),
    tcp_limits,
    log_capacity=64,
  )
  assert_eq(pool.readiness(), LunaOnlineOpenAIConnectionPoolReady)
  let slow = @socket.Tcp::connect(pool.local_addr())
  progress_openai_pool_to_connections(pool, 1, "slow-connect")
  println("openai-pool-stage=slow-connected")
  let slow_request = openai_http_request(
    "/v1/responses", "secret", "{\"model\":\"model-x\",\"input\":\"\\u0001\",\"stream\":true}",
  )
  openai_write_range(slow, slow_request, 0, 1)

  let fast = @socket.Tcp::connect(pool.local_addr())
  progress_openai_pool_to_connections(pool, 2, "fast-connect")
  println("openai-pool-stage=fast-connected")
  let fast_request = openai_http_request(
    "/v1/responses", "secret", "{\"model\":\"model-x\",\"input\":\"\\u0002\",\"stream\":true}",
  )
  openai_write_range(fast, fast_request, 0, fast_request.length())
  // The incomplete first client remains live while the second client crosses
  // admission, the single scheduler owner, physical flight, and full SSE
  // retirement. This is the black-box slow-client isolation proof.
  progress_openai_pool_to_connections(pool, 1, "fast-retire")
  println("openai-pool-stage=fast-retired")
  let response = openai_read_all(fast)
  assert_true(response.contains("HTTP/1.1 200"))
  assert_true(response.contains("response.completed"))
  assert_true(response.contains("data: [DONE]"))
  fast.close()

  // Reuse the retired slot while the first generation is still occupied.
  let reused = @socket.Tcp::connect(pool.local_addr())
  progress_openai_pool_to_connections(pool, 2, "reuse-connect")
  println("openai-pool-stage=reused-connected")
  let rejected = openai_http_request(
    "/v1/responses", "wrong", "{\"model\":\"model-x\",\"input\":\"x\",\"stream\":true}",
  )
  openai_write_range(reused, rejected, 0, rejected.length())
  slow.close()
  progress_openai_pool_to_connections(pool, 0, "all-retire")
  println("openai-pool-stage=all-retired")
  assert_openai_error_response(openai_read_all(reused), "401")
  reused.close()
  let metrics = pool.metrics_snapshot()
  assert_eq(metrics.counter_value(Admissions), 1UL)
  assert_eq(metrics.counter_value(Completions), 1UL)
  assert_eq(metrics.counter_value(NetworkAccepts), 3UL)
  assert_eq(metrics.counter_value(NetworkDisconnects), 3UL)
  assert_eq(metrics.gauge_value(QueueDepth), 0)
  assert_eq(metrics.gauge_value(ActiveRequests), 0)
  drain_openai_pool(pool)
  println("openai-pool-stage=drained")
  assert_eq(pool.state(), LunaOnlineOpenAIConnectionPoolClosed)
  assert_eq(pool.readiness(), LunaOnlineOpenAIConnectionPoolNotReady)
}
