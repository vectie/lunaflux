///|
fn LunaOnlineOpenAIServer::service(
  self : LunaOnlineOpenAIServer,
) -> @online_session.LunaOnlineFramedService {
  if self.services.length() != 1 {
    abort("Luna OpenAI server service invariant")
  }
  self.services[0]
}

///|
fn LunaOnlineOpenAIServer::stream(
  self : LunaOnlineOpenAIServer,
) -> @online_session.LunaOnlineFramedStream {
  if self.streams.length() != 1 {
    abort("Luna OpenAI server stream invariant")
  }
  self.streams[0]
}

///|
fn LunaOnlineOpenAIServer::latch_failure(
  self : LunaOnlineOpenAIServer,
  rule : LunaOnlineTcpRule,
) -> Unit {
  if self.failure_rule == 0 {
    self.failure_rule = luna_online_tcp_rule_code(rule)
  }
}

///|
fn LunaOnlineOpenAIServer::clear_receipt(self : LunaOnlineOpenAIServer) -> Unit {
  if self.receipts.length() == 1 {
    self.receipts[0].abort() catch {
      _ => ()
    }
    self.receipts.clear()
  }
}

///|
fn LunaOnlineOpenAIServer::clear_http(self : LunaOnlineOpenAIServer) -> Unit {
  if self.http_works.length() == 1 {
    self.http_works[0].abort() catch {
      _ => ()
    }
    self.http_works.clear()
  }
  if self.inbound_works.length() == 1 {
    self.inbound_works[0].abort() catch {
      _ => ()
    }
    self.inbound_works.clear()
  }
  if self.inbound_views.length() == 1 {
    self.inbound_views[0].release() catch {
      _ => ()
    }
    self.inbound_views.clear()
  }
}

///|
fn LunaOnlineOpenAIServer::clear_output(self : LunaOnlineOpenAIServer) -> Unit {
  if self.flights.length() == 1 {
    self.flights[0].release() catch {
      _ => ()
    }
    self.flights.clear()
  }
  if self.response_works.length() == 1 {
    self.response_works[0].abort() catch {
      _ => ()
    }
    self.response_works.clear()
  }
  if self.response_views.length() == 1 {
    self.response_views[0].release() catch {
      _ => ()
    }
    self.response_views.clear()
  }
  if self.outbound_works.length() == 1 {
    self.outbound_works[0].abort() catch {
      _ => ()
    }
    self.outbound_works.clear()
  }
  if self.outbound_views.length() == 1 {
    self.outbound_views[0].release() catch {
      _ => ()
    }
    self.outbound_views.clear()
  }
  self.semantic_events.clear()
  self.output_source = 0
  self.output_cursor = 0
  self.output_length = 0
}

///|
fn LunaOnlineOpenAIServer::reset_connection(
  self : LunaOnlineOpenAIServer,
) -> Unit {
  if self.observations.length() != 0 || self.observation_recorded {
    abort("Luna OpenAI observation retirement invariant")
  }
  self.clear_receipt()
  self.clear_http()
  self.clear_output()
  self.embedded_control.clear()
  self.rejections.clear()
  self.streams.clear()
  self.connection_phase = LUNA_OPENAI_CONNECTION_NONE
  self.route_mode = 0
  self.tail_start = 0
  self.tail_length = 0
  self.response_kind = 0
  self.event_terminal = false
  self.terminal_observed = false
  self.sse_started = false
  self.backpressure_episode_active = false
  self.connection_telemetry_active = false
  self.disconnect_pending = false
  self.disconnect_progress_pending = false
}

///|
fn LunaOnlineOpenAIServer::close_listener_on_reactor(
  self : LunaOnlineOpenAIServer,
) -> Unit {
  if self.listeners.length() == 1 {
    self.listeners[0].close()
    self.listeners.clear()
  }
  self.close_listener_pending = false
}

///|
fn LunaOnlineOpenAIServer::begin_retirement(
  self : LunaOnlineOpenAIServer,
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
  self.clear_receipt()
  self.clear_http()
  self.clear_output()
  self.backpressure_episode_active = false
  if self.streams.length() == 1 {
    self.disconnect_pending = true
    try self.stream().disconnect() catch {
      _ => self.disconnect_progress_pending = true
    } noraise {
      _ => self.disconnect_pending = false
    }
  }
  if self.connection_telemetry_active {
    self.record_network_disconnect(reason)
    self.connection_telemetry_active = false
  }
  self.phase = LUNA_ONLINE_TCP_SERVER_RETIRING
}

///|
fn LunaOnlineOpenAIServer::latch_service_drain(
  self : LunaOnlineOpenAIServer,
) -> Unit {
  self.drain_requested = true
  self.close_listener_pending = self.listeners.length() == 1
  if !self.close_listener_pending {
    self.phase = LUNA_ONLINE_TCP_SERVER_DRAINING
  }
}

///|
fn LunaOnlineOpenAIServer::begin_service_drain_if_ready(
  self : LunaOnlineOpenAIServer,
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
pub fn LunaOnlineOpenAIServer::local_addr(
  self : LunaOnlineOpenAIServer,
) -> @socket.Addr {
  self.bound_addr
}

///|
pub fn LunaOnlineOpenAIServer::state(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerState {
  match self.phase {
    0 => LunaOnlineTcpServerListening
    1 => LunaOnlineTcpServerAccepting
    2 => LunaOnlineTcpServerConnected
    3 => LunaOnlineTcpServerRetiring
    4 => LunaOnlineTcpServerDraining
    5 => LunaOnlineTcpServerClosed
    _ => abort("Luna OpenAI server phase invariant")
  }
}

///|
pub fn LunaOnlineOpenAIServer::begin_drain(
  self : LunaOnlineOpenAIServer,
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
pub fn LunaOnlineOpenAIServer::request_cancel(
  self : LunaOnlineOpenAIServer,
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
