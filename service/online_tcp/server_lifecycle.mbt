///|
fn LunaOnlineTcpServer::service(
  self : LunaOnlineTcpServer,
) -> @online_session.LunaOnlineFramedService {
  if self.services.length() != 1 {
    abort("Luna online TCP server service invariant")
  }
  self.services[0]
}

///|
fn LunaOnlineTcpServer::stream(
  self : LunaOnlineTcpServer,
) -> @online_session.LunaOnlineFramedStream {
  if self.streams.length() != 1 {
    abort("Luna online TCP server stream invariant")
  }
  self.streams[0]
}

///|
fn LunaOnlineTcpServer::latch_failure(
  self : LunaOnlineTcpServer,
  rule : LunaOnlineTcpRule,
) -> Unit {
  if self.failure_rule == 0 {
    self.failure_rule = luna_online_tcp_rule_code(rule)
  }
}

///|
fn LunaOnlineTcpServer::release_revoked_output(
  self : LunaOnlineTcpServer,
) -> Unit {
  self.offers.clear()
  if self.flights.length() == 1 {
    try! self.flights[0].release()
    self.flights.clear()
  }
}

///|
fn LunaOnlineTcpServer::reset_connection_scalars(
  self : LunaOnlineTcpServer,
) -> Unit {
  if self.observations.length() != 0 || self.observation_recorded {
    abort("Luna online TCP observation retirement invariant")
  }
  self.streams.clear()
  self.tail_start = 0
  self.tail_length = 0
  self.tail_backpressured = false
  self.request_receipt_complete = false
  self.pipeline_receipt_observed = false
  self.pipeline_receipts_pending = 0
  self.backpressure_episode_active = false
  self.terminal_observed = false
  self.connection_telemetry_active = false
  self.observation_recorded = false
  self.disconnect_pending = false
  self.disconnect_progress_pending = false
  self.offers.clear()
  if self.flights.length() == 1 {
    try! self.flights[0].release()
    self.flights.clear()
  }
}

///|
fn LunaOnlineTcpServer::close_listener_on_reactor(
  self : LunaOnlineTcpServer,
) -> Unit {
  if self.listeners.length() == 1 {
    self.listeners[0].close()
    self.listeners.clear()
  }
  self.close_listener_pending = false
}

///|
fn LunaOnlineTcpServer::begin_connection_retirement(
  self : LunaOnlineTcpServer,
  reason : @instance_log.LunaInstanceLogReason,
  failure? : LunaOnlineTcpRule,
) -> Unit {
  if failure is Some(rule) {
    self.latch_failure(rule)
  }
  if self.connections.length() == 1 {
    self.connections[0].close()
    self.connections.clear()
  }
  self.tail_start = 0
  self.tail_length = 0
  self.tail_backpressured = false
  self.backpressure_episode_active = false
  if self.streams.length() == 1 {
    self.disconnect_pending = true
    self.disconnect_progress_pending = false
    try self.streams[0].disconnect() catch {
      _ => {
        self.latch_failure(LunaOnlineTcpCoordinator)
        self.disconnect_progress_pending = true
      }
    } noraise {
      _ => {
        self.disconnect_pending = false
        self.disconnect_progress_pending = false
      }
    }
  }
  self.release_revoked_output()
  if self.connection_telemetry_active {
    self.record_network_disconnect(reason)
    self.connection_telemetry_active = false
  }
  self.phase = LUNA_ONLINE_TCP_SERVER_RETIRING
}

///|
fn LunaOnlineTcpServer::begin_service_drain_if_ready(
  self : LunaOnlineTcpServer,
) -> Unit {
  if self.service_drain_started || self.listeners.length() != 0 {
    return
  }
  if self.streams.length() == 0 {
    self.service().begin_drain()
    self.service_drain_started = true
    self.phase = LUNA_ONLINE_TCP_SERVER_DRAINING
  }
}

///|
fn LunaOnlineTcpServer::latch_service_drain(self : LunaOnlineTcpServer) -> Unit {
  self.drain_requested = true
  self.close_listener_pending = self.listeners.length() == 1
  if !self.close_listener_pending {
    self.phase = LUNA_ONLINE_TCP_SERVER_DRAINING
  }
}

///|
fn LunaOnlineTcpServer::bounded_transport_wait(
  self : LunaOnlineTcpServer,
  configured_millis : Int,
) -> Int raise LunaOnlineTcpError {
  let remaining = self.stream().transport_wait_remaining_millis() catch {
      _ => raise luna_online_tcp_error(LunaOnlineTcpCoordinator)
    }
  luna_online_tcp_safe_wait(configured_millis, remaining)
}

///|
pub fn LunaOnlineTcpServer::local_addr(
  self : LunaOnlineTcpServer,
) -> @socket.Addr {
  self.bound_addr
}

///|
pub fn LunaOnlineTcpServer::state(
  self : LunaOnlineTcpServer,
) -> LunaOnlineTcpServerState {
  match self.phase {
    0 => LunaOnlineTcpServerListening
    1 => LunaOnlineTcpServerAccepting
    2 => LunaOnlineTcpServerConnected
    3 => LunaOnlineTcpServerRetiring
    4 => LunaOnlineTcpServerDraining
    5 => LunaOnlineTcpServerClosed
    _ => abort("Luna online TCP server phase invariant")
  }
}

///|
pub fn LunaOnlineTcpServer::begin_drain(
  self : LunaOnlineTcpServer,
) -> Unit raise LunaOnlineTcpError {
  if self.activity.reactor_active || self.activity.maintenance_active {
    raise luna_online_tcp_error(LunaOnlineTcpOwnership)
  }
  if self.phase == LUNA_ONLINE_TCP_SERVER_CLOSED || self.drain_requested {
    return
  }
  self.drain_requested = true
  self.close_listener_pending = self.listeners.length() == 1
}

///|
pub fn LunaOnlineTcpServer::request_cancel(
  self : LunaOnlineTcpServer,
) -> Unit raise LunaOnlineTcpError {
  if self.activity.reactor_active || self.activity.maintenance_active {
    raise luna_online_tcp_error(LunaOnlineTcpOwnership)
  }
  if self.phase != LUNA_ONLINE_TCP_SERVER_CONNECTED ||
    self.streams.length() != 1 {
    raise luna_online_tcp_error(LunaOnlineTcpLifecycle)
  }
  self.stream().request_cancel() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpCoordinator)
  }
}
