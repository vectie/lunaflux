///|
fn luna_online_control_health_available(
  failure_rule : Int,
  closed : Bool,
) -> Bool {
  failure_rule == 0 && !closed
}

///|
fn luna_online_control_readiness_available(
  failure_rule : Int,
  closed : Bool,
  drain_requested : Bool,
  listener_count : Int,
  draining : Bool,
  service_ready : Bool,
) -> Bool {
  luna_online_control_health_available(failure_rule, closed) &&
  !drain_requested &&
  listener_count == 1 &&
  !draining &&
  service_ready
}

///|
fn LunaOnlineControlSource::health_available(
  self : LunaOnlineControlSource,
) -> Bool {
  match self {
    LunaNativeControlSource(server) =>
      luna_online_control_health_available(
        server.failure_rule,
        server.phase == LUNA_ONLINE_TCP_SERVER_CLOSED,
      )
    LunaNativePoolControlSource(pool) =>
      pool.health() == LunaOnlineTcpServerHealthy
    LunaOpenAIControlSource(server) =>
      luna_online_control_health_available(
        server.failure_rule,
        server.phase == LUNA_ONLINE_TCP_SERVER_CLOSED,
      )
    LunaOpenAIPoolControlSource(pool) =>
      pool.health() == LunaOnlineTcpServerHealthy
  }
}

///|
fn LunaOnlineControlSource::readiness_available(
  self : LunaOnlineControlSource,
) -> Bool {
  match self {
    LunaNativeControlSource(server) =>
      luna_online_control_readiness_available(
        server.failure_rule,
        server.phase == LUNA_ONLINE_TCP_SERVER_CLOSED,
        server.drain_requested,
        server.listeners.length(),
        server.phase == LUNA_ONLINE_TCP_SERVER_DRAINING,
        server.service().readiness() == LunaOnlineFramedServiceReady,
      )
    LunaNativePoolControlSource(pool) =>
      pool.readiness() == LunaOnlineFramedConnectionPoolReady
    LunaOpenAIControlSource(server) =>
      luna_online_control_readiness_available(
        server.failure_rule,
        server.phase == LUNA_ONLINE_TCP_SERVER_CLOSED,
        server.drain_requested,
        server.listeners.length(),
        server.phase == LUNA_ONLINE_TCP_SERVER_DRAINING,
        server.service().readiness() == LunaOnlineFramedServiceReady,
      )
    LunaOpenAIPoolControlSource(pool) =>
      pool.readiness() == LunaOnlineOpenAIConnectionPoolReady
  }
}

///|
fn LunaOnlineControlSource::metrics_snapshot(
  self : LunaOnlineControlSource,
) -> @metrics.LunaInstanceMetricsSnapshot raise LunaOnlineTcpError {
  match self {
    LunaNativeControlSource(server) => server.metrics_snapshot()
    LunaNativePoolControlSource(pool) => pool.metrics_snapshot()
    LunaOpenAIControlSource(server) => server.metrics_snapshot()
    LunaOpenAIPoolControlSource(pool) => pool.metrics_snapshot()
  }
}

///|
fn LunaOnlineHttpControlServer::source_health(
  self : LunaOnlineHttpControlServer,
) -> Bool {
  !self.failed &&
  self.phase != LUNA_CONTROL_SERVER_CLOSED &&
  self.source.health_available()
}

///|
fn LunaOnlineHttpControlServer::source_readiness(
  self : LunaOnlineHttpControlServer,
) -> Bool {
  self.source_health() &&
  self.production_ready &&
  !self.drain_requested &&
  self.listeners.length() == 1 &&
  self.source.readiness_available()
}

///|
pub fn LunaOnlineHttpControlServer::health(
  self : LunaOnlineHttpControlServer,
) -> LunaOnlineTcpServerHealth {
  if self.source_health() {
    LunaOnlineTcpServerHealthy
  } else {
    LunaOnlineTcpServerFailed
  }
}

///|
pub fn LunaOnlineHttpControlServer::readiness(
  self : LunaOnlineHttpControlServer,
) -> LunaOnlineHttpControlReadiness {
  if self.source_readiness() {
    LunaOnlineHttpControlReady
  } else {
    LunaOnlineHttpControlNotReady
  }
}

///|
pub fn LunaOnlineHttpControlServer::state(
  self : LunaOnlineHttpControlServer,
) -> LunaOnlineHttpControlState {
  match self.phase {
    0 | 1 => LunaOnlineHttpControlListening
    2 | 3 => LunaOnlineHttpControlConnected
    4 => LunaOnlineHttpControlDraining
    5 => LunaOnlineHttpControlClosed
    _ => LunaOnlineHttpControlFailed
  }
}

///|
pub fn LunaOnlineHttpControlServer::local_addr(
  self : LunaOnlineHttpControlServer,
) -> @socket.Addr {
  self.bound_addr
}
