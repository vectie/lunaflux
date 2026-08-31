///|
fn spawned_serving_metric_fixture(
  disconnects? : UInt64 = 0UL,
  kv_pages_free? : Int = 32,
  failures? : UInt64 = 0UL,
) -> SpawnedServingMetricSample {
  {
    admissions: 1UL,
    completions: 1UL,
    cancellations: 0UL,
    deadlines: 0UL,
    failures,
    prompt_tokens: 1UL,
    generated_tokens: 2UL,
    worker_failures: 0UL,
    network_accepts: 1UL,
    network_disconnects: disconnects,
    network_rejections: 0UL,
    queue_depth: 0,
    active_requests: 0,
    kv_pages_used: 0,
    kv_pages_free,
  }
}

///|
fn spawned_serving_wb_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=8,
    max_new_tokens=8,
    max_context_tokens=8,
    max_token_id=2999,
    max_stop_token_ids=2,
    max_stop_strings=2,
    max_stop_string_bytes=16,
    max_trace_bytes=16,
    max_cache_scope_bytes=16,
    max_decoded_delta_bytes=32,
    max_deadline_millis=5000UL,
    max_top_k=3000,
    max_temperature=2.0,
  )
}

///|
fn spawned_serving_wb_request(
  stream_preference? : @inference.StreamPreference = Streaming,
  cache_permission? : @inference.CachePermission = Disabled,
  sampling? : @inference.SamplingParameters = @inference.SamplingParameters::greedy(),
) -> @inference.GenerateRequest raise {
  let limits = spawned_serving_wb_limits()
  @inference.GenerateRequest::new(
    protocol_version=@inference.ProtocolVersion::v1(),
    request_id=@inference.RequestId::new(1UL),
    model_identity=spawned_validation_wb_identity(),
    input=@inference.Input::from_token_ids([1], limits),
    max_new_tokens=2,
    context_ceiling=8,
    sampling~,
    sampling_seed=@inference.SamplingSeed::new(7UL),
    stops=@inference.StopConditions::new([], [], limits),
    stream_preference~,
    deadline=@inference.DeadlineBudget::from_milliseconds(5000UL, limits),
    cache=@inference.CachePolicy::new(
      @inference.CacheScope::from_ascii(b"serving", limits),
      cache_permission,
    ),
    trace=None,
    limits~,
  )
}

///|
test "spawned serving health and readiness evidence is phase exact" {
  assert_true(
    spawned_serving_prebind_state_valid(
      HealthyNotReady,
      RuntimeHealthy,
      RuntimeNotReady,
    ),
  )
  assert_false(
    spawned_serving_prebind_state_valid(Ready, RuntimeHealthy, RuntimeNotReady),
  )
  assert_true(
    spawned_serving_bound_state_valid(Ready, RuntimeHealthy, RuntimeReady),
  )
  assert_false(
    spawned_serving_bound_state_valid(Ready, RuntimeHealthy, RuntimeNotReady),
  )
  assert_true(
    spawned_serving_live_state_valid(
      Ready,
      RuntimeHealthy,
      RuntimeReady,
      LunaOnlineTcpServerListening,
    ),
  )
  assert_true(
    spawned_serving_live_state_valid(
      Ready,
      RuntimeHealthy,
      RuntimeNotReady,
      LunaOnlineTcpServerConnected,
    ),
  )
  assert_false(
    spawned_serving_live_state_valid(
      Ready,
      RuntimeHealthy,
      RuntimeReady,
      LunaOnlineTcpServerConnected,
    ),
  )
  assert_false(
    spawned_serving_live_state_valid(
      Ready,
      RuntimeHealthy,
      RuntimeNotReady,
      LunaOnlineTcpServerListening,
    ),
  )
  assert_true(
    spawned_serving_live_state_valid(
      Ready,
      RuntimeHealthy,
      RuntimeReady,
      LunaOnlineTcpServerAccepting,
    ),
  )
  assert_false(
    spawned_serving_live_state_valid(
      Ready,
      RuntimeHealthy,
      RuntimeNotReady,
      LunaOnlineTcpServerAccepting,
    ),
  )
  assert_true(
    spawned_serving_live_state_valid(
      Ready,
      RuntimeHealthy,
      RuntimeReady,
      LunaOnlineTcpServerRetiring,
    ),
  )
  assert_true(
    spawned_serving_live_state_valid(
      Ready,
      RuntimeHealthy,
      RuntimeNotReady,
      LunaOnlineTcpServerRetiring,
    ),
  )
  assert_false(
    spawned_serving_live_state_valid(
      Ready,
      RuntimeHealthy,
      RuntimeNotReady,
      LunaOnlineTcpServerDraining,
    ),
  )
  assert_false(
    spawned_serving_live_state_valid(
      Ready,
      RuntimeHealthy,
      RuntimeNotReady,
      LunaOnlineTcpServerClosed,
    ),
  )
  assert_false(
    spawned_serving_live_state_valid(
      Draining,
      RuntimeHealthy,
      RuntimeNotReady,
      LunaOnlineTcpServerRetiring,
    ),
  )
  assert_false(
    spawned_serving_live_state_valid(
      Ready,
      RuntimeUnhealthy,
      RuntimeNotReady,
      LunaOnlineTcpServerRetiring,
    ),
  )
  assert_true(
    spawned_serving_closed_state_valid(
      Closed,
      RuntimeClosed,
      RuntimeNotReady,
      true,
    ),
  )
  assert_false(
    spawned_serving_closed_state_valid(
      Closed,
      RuntimeClosed,
      RuntimeReady,
      true,
    ),
  )
}

///|
test "spawned serving admits exactly one native server or framed pool" {
  assert_true(spawned_serving_ingress_shape_valid(1, 0, 0, 0))
  assert_true(spawned_serving_ingress_shape_valid(0, 1, 0, 0))
  assert_false(spawned_serving_ingress_shape_valid(1, 1, 0, 0))
  assert_false(spawned_serving_ingress_shape_valid(0, 0, 0, 0))
  assert_false(spawned_serving_ingress_shape_valid(0, 1, 1, 0))
  assert_false(spawned_serving_ingress_shape_valid(0, 1, 0, 1))
}

///|
test "spawned serving framed pool requires exact live connection balance" {
  assert_true(
    spawned_serving_pool_live_state_valid(
      Ready,
      RuntimeHealthy,
      RuntimeReady,
      LunaOnlineFramedConnectionPoolListening,
      0,
    ),
  )
  assert_true(
    spawned_serving_pool_live_state_valid(
      Ready,
      RuntimeHealthy,
      RuntimeReady,
      LunaOnlineFramedConnectionPoolActive,
      1,
    ),
  )
  assert_false(
    spawned_serving_pool_live_state_valid(
      Ready,
      RuntimeHealthy,
      RuntimeReady,
      LunaOnlineFramedConnectionPoolListening,
      1,
    ),
  )
  assert_false(
    spawned_serving_pool_live_state_valid(
      Ready,
      RuntimeHealthy,
      RuntimeNotReady,
      LunaOnlineFramedConnectionPoolActive,
      1,
    ),
  )
  assert_false(
    spawned_serving_pool_live_state_valid(
      Ready,
      RuntimeHealthy,
      RuntimeReady,
      LunaOnlineFramedConnectionPoolActive,
      0,
    ),
  )
  assert_false(
    spawned_serving_pool_live_state_valid(
      Ready,
      RuntimeHealthy,
      RuntimeReady,
      LunaOnlineFramedConnectionPoolDraining,
      0,
    ),
  )
}

///|
test "spawned serving terminal ingress preserves server and pool lifecycle" {
  assert_true(spawned_serving_terminal_ingress_state_valid(1, 0, true, false))
  assert_true(spawned_serving_terminal_ingress_state_valid(0, 1, false, true))
  assert_false(spawned_serving_terminal_ingress_state_valid(1, 0, false, true))
  assert_false(spawned_serving_terminal_ingress_state_valid(0, 1, true, false))
  assert_false(spawned_serving_terminal_ingress_state_valid(1, 1, true, true))
  assert_false(spawned_serving_terminal_ingress_state_valid(0, 0, false, false))
}

///|
test "spawned serving terminal metrics require exact request and KV balance" {
  assert_true(
    spawned_serving_terminal_metrics_valid(
      spawned_serving_metric_fixture(),
      1,
      32,
      false,
    ),
  )
  assert_true(
    spawned_serving_terminal_metrics_valid(
      spawned_serving_metric_fixture(disconnects=1UL),
      1,
      32,
      true,
    ),
  )
  assert_false(
    spawned_serving_terminal_metrics_valid(
      spawned_serving_metric_fixture(kv_pages_free=31),
      1,
      32,
      false,
    ),
  )
  assert_false(
    spawned_serving_terminal_metrics_valid(
      spawned_serving_metric_fixture(disconnects=1UL),
      1,
      32,
      false,
    ),
  )
  assert_false(
    spawned_serving_terminal_metrics_valid(
      spawned_serving_metric_fixture(failures=1UL),
      1,
      32,
      false,
    ),
  )
}

///|
test "spawned serving contract bounds reject unbounded or cross-page probes" {
  assert_true(
    spawned_serving_contract_scalars_valid(
      128, 1024, 1031, 2185, 2999, 1, 256, 256, 8, 1024, 5000,
    ),
  )
  assert_false(
    spawned_serving_contract_scalars_valid(
      1025, 1024, 1031, 2185, 2999, 1, 256, 256, 8, 1024, 5000,
    ),
  )
  assert_false(
    spawned_serving_contract_scalars_valid(
      128, 1024, 3000, 2185, 2999, 1, 256, 256, 8, 1024, 5000,
    ),
  )
  assert_false(
    spawned_serving_contract_scalars_valid(
      128, 1024, 1031, 2185, 2999, 7, 256, 256, 8, 1024, 5000,
    ),
  )
  assert_false(
    spawned_serving_contract_scalars_valid(
      128, 1024, 1031, 2185, 2999, 1, 256, 256, 8, 0, 5000,
    ),
  )
}

///|
test "spawned serving request rejects buffered cacheable and sampled variants" {
  let identity = spawned_validation_wb_identity()
  assert_true(
    spawned_serving_request_matches(
      spawned_serving_wb_request(),
      identity,
      1,
      8,
    ),
  )
  assert_false(
    spawned_serving_request_matches(
      spawned_serving_wb_request(stream_preference=Buffered),
      identity,
      1,
      8,
    ),
  )
  assert_false(
    spawned_serving_request_matches(
      spawned_serving_wb_request(cache_permission=ReadOnly),
      identity,
      1,
      8,
    ),
  )
  assert_false(
    spawned_serving_request_matches(
      spawned_serving_wb_request(
        sampling=@inference.SamplingParameters::sample(
          temperature=1.0,
          spawned_serving_wb_limits(),
        ),
      ),
      identity,
      1,
      8,
    ),
  )
}

///|
test "spawned serving contract defensively copies the authenticated frame" {
  let limits = spawned_serving_wb_limits()
  let framed = @framed_wire.FramedWireLimits::new(limits, max_frame_bytes=4096)
  let encoded = @framed_wire.RequestFrameBuffer::new(framed).encode(
    spawned_serving_wb_request(),
  )
  let source = FixedArray::make(encoded.length(), b'\x00')
  ignore(encoded.copy_to(source, destination_offset=0))
  let original = source[0]
  let contract = RuntimeSpawnedServingValidationContract::new(
    source,
    spawned_validation_wb_identity(),
    1031,
    2185,
    1,
    8,
    8,
    limits,
    4096,
    1024,
    5000,
  )
  source[0] = source[0] ^ b'\x01'
  assert_eq(contract.frame[0], original)
}

///|
test "spawned serving frame length decode is canonical little endian" {
  let header = FixedArray::make(16, b'\x00')
  header[12] = b'\x34'
  header[13] = b'\x12'
  assert_eq(spawned_serving_read_u32(header, 12), 0x1234)
}
