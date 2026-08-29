///|
fn new_luna_openai_connection_slot(
  inference : @inference.InferenceLimits,
  identity : @spec.ModelIdentity,
  tcp_limits : LunaOnlineTcpLimits,
  pool_limits : LunaOnlineOpenAIConnectionPoolLimits,
  http_limits : @http1.LunaHttp1Limits,
  http_budget : @http1.LunaHttp1StepBudget,
  response_budget : @http1.LunaHttp1ResponseStepBudget,
  auth_policy : @api_auth.LunaApiAuthPolicy,
  inbound_budget : @openai_compat.LunaOpenAICompatStepBudget,
  outbound_budget : @openai_compat.LunaOpenAICompatStepBudget,
  inbound_limits : @openai_compat.LunaOpenAIInboundLimits,
  template : @openai_compat.LunaOpenAIChatTemplate,
  deployment_model_alias : FixedArray[Byte],
  model_alias_offset : Int,
  model_alias_length : Int,
  response_id_prefix : FixedArray[Byte],
  response_id_prefix_offset : Int,
  response_id_prefix_length : Int,
  cache_scope : @inference.CacheScope,
  max_new_tokens : Int,
  context_ceiling : Int,
  sampling : @inference.SamplingParameters,
  sampling_seed : @inference.SamplingSeed,
  deadline : @inference.DeadlineBudget,
) -> LunaOnlineOpenAIConnectionSlot raise LunaOnlineTcpError {
  let receipt_workspace = @request_admission.LunaRequestReceiptWorkspace::new(
    @monotonic_clock.MonotonicClock::system(),
    deadline,
  ) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let http_workspace = @http1.LunaHttp1Workspace::new(
    http_limits, http_budget, auth_policy,
  ) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let inbound_workspace = @openai_compat.LunaOpenAIInboundWorkspace::new(
    inference,
    inbound_budget,
    inbound_limits,
    template,
    deployment_model_alias,
    model_alias_offset~,
    model_alias_length~,
    identity,
    cache_scope,
    max_new_tokens~,
    context_ceiling~,
    sampling,
    sampling_seed,
    deadline,
  ) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let outbound_workspace = @openai_compat.LunaOpenAICompatWorkspace::new(
    inference,
    outbound_budget,
    deployment_model_alias,
    model_alias_offset~,
    model_alias_length~,
    response_id_prefix,
    response_id_prefix_offset~,
    response_id_prefix_length~,
  ) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let embedded_control = new_luna_openai_embedded_control()
  {
    receipt_workspace,
    http_workspace,
    response_workspace: @http1.LunaHttp1ResponseWorkspace::new(response_budget),
    inbound_workspace,
    outbound_workspace,
    embedded_control,
    connections: Array::new(capacity=1),
    receipts: Array::new(capacity=1),
    http_works: Array::new(capacity=1),
    response_works: Array::new(capacity=1),
    response_views: Array::new(capacity=1),
    inbound_works: Array::new(capacity=1),
    inbound_views: Array::new(capacity=1),
    outbound_works: Array::new(capacity=1),
    outbound_views: Array::new(capacity=1),
    semantic_events: Array::new(capacity=1),
    rejections: Array::new(capacity=1),
    routes: Array::new(capacity=1),
    input: FixedArray::make(tcp_limits.read_chunk_bytes, b'\x00'),
    output: LunaOnlineTcpOutputScratch::new(pool_limits.output_buffer_bytes) catch {
      _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
    },
    output_writes: Array::new(capacity=1),
    flights: Array::new(capacity=1),
    outbound_copy_quanta: outbound_budget.work_units(),
    phase: LUNA_OPENAI_POOL_FREE,
    generation: 0UL,
    request_sequence: 0UL,
    route_mode: 0,
    tail_start: 0,
    tail_length: 0,
    response_kind: 0,
    response_copy_cursor: 0,
    compat_copy_cursor: 0,
    output_length: 0,
    output_cursor: 0,
    event_terminal: false,
    terminal_observed: false,
    sse_started: false,
    cancel_requested: false,
  }
}

///|
/// Prepares every connection-owned parser, renderer, and bounded output cell
/// before listener activation. No scheduler or device authority is duplicated.
pub fn prepare_luna_online_openai_connection_pool(
  service : @online_session.LunaOnlineFramedServicePreparation,
  tcp_limits : LunaOnlineTcpLimits,
  pool_limits : LunaOnlineOpenAIConnectionPoolLimits,
  http_limits : @http1.LunaHttp1Limits,
  http_budget : @http1.LunaHttp1StepBudget,
  response_budget : @http1.LunaHttp1ResponseStepBudget,
  auth_policy : @api_auth.LunaApiAuthPolicy,
  inbound_budget : @openai_compat.LunaOpenAICompatStepBudget,
  outbound_budget : @openai_compat.LunaOpenAICompatStepBudget,
  inbound_limits : @openai_compat.LunaOpenAIInboundLimits,
  template : @openai_compat.LunaOpenAIChatTemplate,
  deployment_model_alias : FixedArray[Byte],
  model_alias_offset~ : Int,
  model_alias_length~ : Int,
  response_id_prefix : FixedArray[Byte],
  response_id_prefix_offset~ : Int,
  response_id_prefix_length~ : Int,
  cache_scope : @inference.CacheScope,
  max_new_tokens~ : Int,
  context_ceiling~ : Int,
  sampling : @inference.SamplingParameters,
  sampling_seed : @inference.SamplingSeed,
  deadline : @inference.DeadlineBudget,
  production_ready? : Bool = false,
) -> LunaOnlineOpenAIConnectionPoolPreparation raise LunaOnlineTcpError {
  let transferred = Ref(false)
  defer (if !transferred.val { auth_policy.close() })
  let inference = service.inference_limits() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let identity = service.model_identity() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let maximum_wait_millis = service.maximum_transport_wait_millis() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  tcp_limits.require_fits_transport_wait(maximum_wait_millis) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpLimits)
  }
  if !pool_limits.fits_write_chunk(tcp_limits) ||
    pool_limits.io_poll_timeout_millis >
    Int::min(
      tcp_limits.input_idle_timeout_millis,
      tcp_limits.write_timeout_millis,
    ) {
    raise luna_online_tcp_error(LunaOnlineTcpLimits)
  }
  let slots = FixedArray::makei(pool_limits.connection_capacity, _ => {
    new_luna_openai_connection_slot(
      inference, identity, tcp_limits, pool_limits, http_limits, http_budget, response_budget,
      auth_policy, inbound_budget, outbound_budget, inbound_limits, template, deployment_model_alias,
      model_alias_offset, model_alias_length, response_id_prefix, response_id_prefix_offset,
      response_id_prefix_length, cache_scope, max_new_tokens, context_ceiling, sampling,
      sampling_seed, deadline,
    )
  })
  transferred.val = true
  {
    service,
    slots,
    limits: pool_limits,
    maximum_wait_millis,
    auth_policy,
    production_ready,
    consumed: false,
  }
}

///|
pub async fn bind_luna_online_openai_connection_pool(
  addr : @socket.Addr,
  preparation : LunaOnlineOpenAIConnectionPoolPreparation,
  tcp_limits : LunaOnlineTcpLimits,
  log_capacity~ : Int,
) -> LunaOnlineOpenAIConnectionPool {
  if preparation.consumed {
    raise luna_online_tcp_error(LunaOnlineTcpOwnership)
  }
  let transferred = Ref(false)
  defer (if !transferred.val {
    preparation.auth_policy.close()
    preparation.consumed = true
  })
  tcp_limits.require_fits_transport_wait(preparation.maximum_wait_millis) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpLimits)
  }
  if !preparation.limits.fits_write_chunk(tcp_limits) {
    raise luna_online_tcp_error(LunaOnlineTcpLimits)
  }
  let initial = preparation.service.luna_telemetry() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let initial_plan = preparation.service.luna_plan_telemetry() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let telemetry = LunaServerTelemetryBridge::new(log_capacity)
  if !telemetry.refresh(initial, initial_plan) {
    raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let listener = bind_luna_online_tcp_listener(addr)
  let service = preparation.service.take_ready() catch {
    _ => {
      listener.close()
      raise luna_online_tcp_error(LunaOnlineTcpPreparation)
    }
  }
  let opened = service.open_semantic_stream()
  if opened.kind() != LunaOnlineFramedOpened {
    listener.close()
    service.begin_drain()
    raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let stream = service.take_open_stream(opened) catch {
    _ => {
      listener.close()
      service.begin_drain()
      raise luna_online_tcp_error(LunaOnlineTcpPreparation)
    }
  }
  preparation.consumed = true
  transferred.val = true
  telemetry.record_log(LunaInstanceReady, LunaLogNoReason)
  {
    tcp_limits,
    pool_limits: preparation.limits,
    bound_addr: listener.addr,
    listeners: [listener],
    services: [service],
    streams: [stream],
    slots: preparation.slots,
    auth_policy: preparation.auth_policy,
    telemetry,
    active_connections: 0,
    ingress_slot: -1,
    io_cursor: 0,
    accept_turn: true,
    next_request_sequence: 0UL,
    drain_requested: false,
    stream_disconnect_started: false,
    service_drain_started: false,
    closed: false,
    production_ready: preparation.production_ready,
    cold_start_latency_millis: 0,
  }
}
