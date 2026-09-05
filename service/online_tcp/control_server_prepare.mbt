///|
pub fn prepare_luna_online_http_control_for_native(
  source : LunaOnlineTcpServer,
  limits : @http1.LunaHttp1ControlLimits,
) -> LunaOnlineHttpControlPreparation raise LunaOnlineTcpError {
  let workspace = @http1.LunaHttp1ControlWorkspace::new(limits) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  {
    source: LunaNativeControlSource(source),
    workspace,
    production_ready: true,
    consumed: false,
  }
}

///|
pub fn prepare_luna_online_http_control_for_native_pool(
  source : LunaOnlineFramedConnectionPool,
  limits : @http1.LunaHttp1ControlLimits,
) -> LunaOnlineHttpControlPreparation raise LunaOnlineTcpError {
  let workspace = @http1.LunaHttp1ControlWorkspace::new(limits) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  {
    source: LunaNativePoolControlSource(source),
    workspace,
    production_ready: true,
    consumed: false,
  }
}

///|
fn prepare_luna_online_http_control_for_openai(
  source : LunaOnlineOpenAIServer,
  limits : @http1.LunaHttp1ControlLimits,
  production_ready~ : Bool,
) -> LunaOnlineHttpControlPreparation raise LunaOnlineTcpError {
  let workspace = @http1.LunaHttp1ControlWorkspace::new(limits) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  {
    source: LunaOpenAIControlSource(source),
    workspace,
    production_ready,
    consumed: false,
  }
}

///|
fn prepare_luna_online_http_control_for_openai_pool(
  source : LunaOnlineOpenAIConnectionPool,
  limits : @http1.LunaHttp1ControlLimits,
  production_ready~ : Bool,
) -> LunaOnlineHttpControlPreparation raise LunaOnlineTcpError {
  let workspace = @http1.LunaHttp1ControlWorkspace::new(limits) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  {
    source: LunaOpenAIPoolControlSource(source),
    workspace,
    production_ready,
    consumed: false,
  }
}

///|
pub fn prepare_luna_online_http_control_for_openai_pool_production(
  source : LunaOnlineOpenAIConnectionPool,
  limits : @http1.LunaHttp1ControlLimits,
) -> LunaOnlineHttpControlPreparation raise LunaOnlineTcpError {
  prepare_luna_online_http_control_for_openai_pool(
    source,
    limits,
    production_ready=true,
  )
}

///|
pub fn prepare_luna_online_http_control_for_openai_pool_qualification(
  source : LunaOnlineOpenAIConnectionPool,
  limits : @http1.LunaHttp1ControlLimits,
) -> LunaOnlineHttpControlPreparation raise LunaOnlineTcpError {
  prepare_luna_online_http_control_for_openai_pool(
    source,
    limits,
    production_ready=false,
  )
}

///|
pub fn prepare_luna_online_http_control_for_openai_production(
  source : LunaOnlineOpenAIServer,
  limits : @http1.LunaHttp1ControlLimits,
) -> LunaOnlineHttpControlPreparation raise LunaOnlineTcpError {
  prepare_luna_online_http_control_for_openai(
    source,
    limits,
    production_ready=true,
  )
}

///|
pub fn prepare_luna_online_http_control_for_openai_qualification(
  source : LunaOnlineOpenAIServer,
  limits : @http1.LunaHttp1ControlLimits,
) -> LunaOnlineHttpControlPreparation raise LunaOnlineTcpError {
  prepare_luna_online_http_control_for_openai(
    source,
    limits,
    production_ready=false,
  )
}

///|
fn luna_online_control_response_bytes(
  kind : @http1.LunaHttp1ControlResponseKind,
) -> Bytes raise LunaOnlineTcpError {
  let response = @http1.LunaHttp1ControlResponse::new(kind)
  let storage = FixedArray::make(response.length(), b'\x00')
  let copied = response.copy_chunk(
    storage,
    destination_offset=0,
    source_offset=0,
    length=storage.length(),
  ) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  if copied != storage.length() {
    raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  Bytes::makei(storage.length(), index => storage[index])
}

///|
fn luna_online_control_responses() -> FixedArray[Bytes] raise LunaOnlineTcpError {
  [
    luna_online_control_response_bytes(LunaHttp1ControlHealthy),
    luna_online_control_response_bytes(LunaHttp1ControlUnhealthy),
    luna_online_control_response_bytes(LunaHttp1ControlReadyResponse),
    luna_online_control_response_bytes(LunaHttp1ControlNotReadyResponse),
    luna_online_control_response_bytes(LunaHttp1ControlBadRequest),
  ]
}

///|
fn luna_online_startup_measurement_valid(duration_millis : Int) -> Bool {
  duration_millis >= 0 && duration_millis <= 86400000
}

///|
fn luna_online_startup_measurement_publish_allowed(
  published : Bool,
  failed : Bool,
  drain_requested : Bool,
  phase : Int,
) -> Bool {
  !published &&
  !failed &&
  !drain_requested &&
  phase == LUNA_CONTROL_SERVER_LISTENING
}

///|
/// Publishes the exact startup duration once, before the runtime owner can
/// advertise readiness. The scalar is rendered with the live metrics snapshot
/// only when the control listener accepts an exact `/metrics` request.
pub fn LunaOnlineHttpControlServer::publish_startup_measurement(
  self : LunaOnlineHttpControlServer,
  duration_millis : Int,
) -> Unit raise LunaOnlineTcpError {
  if !luna_online_startup_measurement_valid(duration_millis) ||
    !luna_online_startup_measurement_publish_allowed(
      self.startup_measurement_published,
      self.failed,
      self.drain_requested,
      self.phase,
    ) {
    raise luna_online_tcp_error(LunaOnlineTcpLifecycle)
  }
  self.cold_start_latency_millis = duration_millis
  self.startup_measurement_published = true
}

///|
pub async fn bind_luna_online_http_control_server(
  preparation : LunaOnlineHttpControlPreparation,
  limits : LunaOnlineTcpLimits,
) -> LunaOnlineHttpControlServer {
  if preparation.consumed || !limits.is_valid() {
    raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  // Allocate every request-path backing before the listener becomes
  // authoritative. The runtime owner publishes the bounded startup scalar;
  // live metrics reuse the startup-owned output scratch on control turns.
  let input = FixedArray::make(limits.read_chunk_bytes, b'\x00')
  let responses = luna_online_control_responses()
  let metrics_output = LunaOnlineTcpOutputScratch::new(
    LUNA_ONLINE_CONTROL_METRICS_RESPONSE_CAPACITY,
  ) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let addr = @socket.Addr::parse("127.0.0.1:0") catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpBind)
  }
  let listener = bind_luna_online_tcp_listener(addr)
  if @async.is_being_cancelled() {
    listener.close()
    @async.pause()
    raise luna_online_tcp_error(LunaOnlineTcpCancelled)
  }
  preparation.consumed = true
  {
    source: preparation.source,
    production_ready: preparation.production_ready,
    limits,
    bound_addr: listener.addr,
    listeners: [listener],
    connections: Array::new(capacity=1),
    workspace: preparation.workspace,
    requests: Array::new(capacity=1),
    input,
    responses,
    metrics_output,
    metrics_flights: Array::new(capacity=1),
    output: b"",
    output_cursor: 0,
    output_length: 0,
    phase: LUNA_CONTROL_SERVER_LISTENING,
    drain_requested: false,
    failed: false,
    startup_measurement_published: false,
    cold_start_latency_millis: -1,
    io_started_millis: 0UL,
  }
}
