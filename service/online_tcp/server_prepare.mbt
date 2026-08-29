///|
async fn bind_luna_online_tcp_server_mode(
  addr : @socket.Addr,
  preparation : @online_session.LunaOnlineFramedServicePreparation,
  limits : LunaOnlineTcpLimits,
  log_capacity~ : Int,
  pipeline_mode~ : Bool,
) -> LunaOnlineTcpServer {
  if !limits.is_valid() {
    raise luna_online_tcp_error(LunaOnlineTcpLimits)
  }
  let maximum_wait = preparation.maximum_transport_wait_millis() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  if !limits.fits_transport_wait(maximum_wait) {
    raise luna_online_tcp_error(LunaOnlineTcpLimits)
  }
  let initial = preparation.luna_telemetry() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let initial_plan = preparation.luna_plan_telemetry() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let input = FixedArray::make(limits.read_chunk_bytes, b'\x00')
  let output = LunaOnlineTcpOutputScratch::new(limits.write_chunk_bytes) catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpLimits)
  }
  let telemetry = LunaServerTelemetryBridge::new(log_capacity)
  if !telemetry.refresh(initial, initial_plan) {
    raise luna_online_tcp_error(LunaOnlineTcpPreparation)
  }
  let owner : LunaOnlineTcpServer = {
    limits,
    bound_addr: addr,
    listeners: Array::new(capacity=1),
    connections: Array::new(capacity=1),
    services: Array::new(capacity=1),
    streams: Array::new(capacity=1),
    input,
    output,
    offers: Array::new(capacity=1),
    flights: Array::new(capacity=1),
    observations: Array::new(capacity=1),
    telemetry,
    activity: { reactor_active: false, maintenance_active: false },
    pipeline_mode,
    phase: LUNA_ONLINE_TCP_SERVER_LISTENING,
    tail_start: 0,
    tail_length: 0,
    tail_backpressured: false,
    request_receipt_complete: false,
    pipeline_receipt_observed: false,
    pipeline_receipts_pending: 0,
    backpressure_episode_active: false,
    terminal_observed: false,
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
  }
  let listener = bind_luna_online_tcp_listener(addr)
  if @async.is_being_cancelled() {
    listener.close()
    @async.pause()
    raise luna_online_tcp_error(LunaOnlineTcpCancelled)
  }
  owner.bound_addr = listener.addr
  owner.listeners.push(listener)
  let service = preparation.take_ready() catch {
    _ => {
      listener.close()
      owner.listeners.clear()
      raise luna_online_tcp_error(LunaOnlineTcpPreparation)
    }
  }
  owner.services.push(service)
  owner.record_log(LunaInstanceReady, LunaLogNoReason)
  owner
}

///|
/// Binds the max-one compatibility server. A terminal request retires the
/// active connection before the persistent listener accepts another peer.
pub async fn bind_luna_online_tcp_server(
  addr : @socket.Addr,
  preparation : @online_session.LunaOnlineFramedServicePreparation,
  limits : LunaOnlineTcpLimits,
  log_capacity~ : Int,
) -> LunaOnlineTcpServer {
  bind_luna_online_tcp_server_mode(
    addr,
    preparation,
    limits,
    log_capacity~,
    pipeline_mode=false,
  )
}

///|
/// Binds one native framed connection that may pipeline bounded request
/// frames through the same authenticated Service Stream. Events remain in the
/// Service's global order; a typed rejection or peer disconnect is terminal
/// for the connection.
pub async fn bind_luna_online_tcp_pipeline_server(
  addr : @socket.Addr,
  preparation : @online_session.LunaOnlineFramedServicePreparation,
  limits : LunaOnlineTcpLimits,
  log_capacity~ : Int,
) -> LunaOnlineTcpServer {
  bind_luna_online_tcp_server_mode(
    addr,
    preparation,
    limits,
    log_capacity~,
    pipeline_mode=true,
  )
}
