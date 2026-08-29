///|
pub fn prepare_luna_online_openai_server(
  service : @online_session.LunaOnlineFramedServicePreparation,
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
) -> LunaOnlineOpenAIServerPreparation raise LunaOnlineTcpError {
  let transferred = Ref(false)
  defer (if !transferred.val { auth_policy.close() })
  let inference = service.inference_limits() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let identity = service.model_identity() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let maximum_wait = service.maximum_transport_wait_millis() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let deadline_millis = deadline.milliseconds()
  if deadline_millis == 0UL || deadline_millis > 0x7fffffffUL {
    raise luna_online_tcp_error(LunaOnlineTcpLimits)
  }
  let receipt = @request_admission.LunaRequestReceiptWorkspace::new(
    @monotonic_clock.MonotonicClock::system(),
    deadline,
  ) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let http = @http1.LunaHttp1Workspace::new(
    http_limits, http_budget, auth_policy,
  ) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let response = @http1.LunaHttp1ResponseWorkspace::new(response_budget)
  let inbound = @openai_compat.LunaOpenAIInboundWorkspace::new(
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
  let outbound = @openai_compat.LunaOpenAICompatWorkspace::new(
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
  transferred.val = true
  {
    service,
    receipt,
    http,
    response,
    inbound,
    outbound,
    embedded_control,
    production_ready,
    auth_policy,
    consumed: false,
    maximum_wait_millis: Int::min(maximum_wait, deadline_millis.to_int()),
    response_copy_quanta: response_budget.work_units(),
    outbound_copy_quanta: outbound_budget.work_units(),
  }
}

///|
pub async fn bind_luna_online_openai_server(
  addr : @socket.Addr,
  preparation : LunaOnlineOpenAIServerPreparation,
  limits : LunaOnlineTcpLimits,
  log_capacity~ : Int,
) -> LunaOnlineOpenAIServer {
  if preparation.consumed {
    raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let transferred = Ref(false)
  defer (if !transferred.val {
    preparation.auth_policy.close()
    preparation.consumed = true
  })
  if !limits.is_valid() ||
    !limits.fits_transport_wait(preparation.maximum_wait_millis) {
    raise luna_online_tcp_error(LunaOnlineTcpPreparation)
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
  let input = FixedArray::make(limits.read_chunk_bytes, b'\x00')
  let output = LunaOnlineTcpOutputScratch::new(limits.write_chunk_bytes) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpLimits)
  }
  let listener = bind_luna_online_tcp_listener(addr)
  if @async.is_being_cancelled() {
    listener.close()
    @async.pause()
    raise luna_online_tcp_error(LunaOnlineTcpCancelled)
  }
  let service = preparation.service.take_ready() catch {
    _ => {
      listener.close()
      raise luna_online_tcp_error(LunaOnlineTcpPreparation)
    }
  }
  preparation.consumed = true
  transferred.val = true
  let owner : LunaOnlineOpenAIServer = {
    limits,
    bound_addr: listener.addr,
    listeners: [listener],
    connections: Array::new(capacity=1),
    services: [service],
    streams: Array::new(capacity=1),
    receipt_workspace: preparation.receipt,
    receipts: Array::new(capacity=1),
    http_workspace: preparation.http,
    http_works: Array::new(capacity=1),
    response_workspace: preparation.response,
    response_works: Array::new(capacity=1),
    response_views: Array::new(capacity=1),
    inbound_workspace: preparation.inbound,
    inbound_works: Array::new(capacity=1),
    inbound_views: Array::new(capacity=1),
    outbound_workspace: preparation.outbound,
    auth_policy: preparation.auth_policy,
    outbound_works: Array::new(capacity=1),
    outbound_views: Array::new(capacity=1),
    embedded_control: preparation.embedded_control,
    semantic_events: Array::new(capacity=1),
    rejections: Array::new(capacity=1),
    observations: Array::new(capacity=1),
    input,
    output,
    flights: Array::new(capacity=1),
    telemetry,
    activity: { reactor_active: false, maintenance_active: false },
    response_copy_quanta: preparation.response_copy_quanta,
    outbound_copy_quanta: preparation.outbound_copy_quanta,
    phase: LUNA_ONLINE_TCP_SERVER_LISTENING,
    connection_phase: LUNA_OPENAI_CONNECTION_NONE,
    connection_epoch: 0UL,
    request_sequence: 0UL,
    route_mode: 0,
    tail_start: 0,
    tail_length: 0,
    response_kind: 0,
    output_source: 0,
    output_cursor: 0,
    output_length: 0,
    event_terminal: false,
    terminal_observed: false,
    sse_started: false,
    backpressure_episode_active: false,
    connection_telemetry_active: false,
    observation_recorded: false,
    disconnect_pending: false,
    disconnect_progress_pending: false,
    drain_requested: false,
    close_listener_pending: false,
    service_drain_started: false,
    drain_logged: false,
    closed_logged: false,
    failure_rule: 0,
    production_ready: preparation.production_ready,
    cold_start_latency_millis: 0,
  }
  owner.record_log(LunaInstanceReady, LunaLogNoReason)
  owner
}
