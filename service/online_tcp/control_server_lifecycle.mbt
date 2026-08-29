///|
fn LunaOnlineHttpControlServer::clear_request(
  self : LunaOnlineHttpControlServer,
) -> Unit {
  if self.requests.length() == 1 {
    self.requests[0].abort() catch {
      _ => ()
    }
    self.requests.clear()
  }
}

///|
fn LunaOnlineHttpControlServer::clear_metrics_flight(
  self : LunaOnlineHttpControlServer,
) -> Bool {
  if self.metrics_flights.length() == 0 {
    return self.metrics_output.is_idle()
  }
  if self.metrics_flights.length() != 1 {
    return false
  }
  self.metrics_flights[0].release() catch {
    _ => return false
  }
  self.metrics_flights.clear()
  true
}

///|
fn LunaOnlineHttpControlServer::retire_connection(
  self : LunaOnlineHttpControlServer,
) -> Unit {
  if self.connections.length() == 1 {
    self.connections[0].close()
    self.connections.clear()
  }
  self.clear_request()
  let metrics_cleared = self.clear_metrics_flight()
  self.output = b""
  self.output_cursor = 0
  self.output_length = 0
  if !metrics_cleared {
    self.failed = true
    self.drain_requested = true
  }
  self.phase = if self.failed {
    LUNA_CONTROL_SERVER_FAILED
  } else if self.drain_requested {
    LUNA_CONTROL_SERVER_DRAINING
  } else {
    LUNA_CONTROL_SERVER_LISTENING
  }
}

///|
fn LunaOnlineHttpControlServer::fail(
  self : LunaOnlineHttpControlServer,
) -> Unit {
  self.failed = true
  self.drain_requested = true
  if self.listeners.length() == 1 {
    self.listeners[0].close()
    self.listeners.clear()
  }
  if self.connections.length() == 1 {
    self.connections[0].close()
    self.connections.clear()
  }
  self.clear_request()
  ignore(self.clear_metrics_flight())
  self.output = b""
  self.output_cursor = 0
  self.output_length = 0
  self.phase = LUNA_CONTROL_SERVER_FAILED
}

///|
pub fn LunaOnlineHttpControlServer::begin_drain(
  self : LunaOnlineHttpControlServer,
) -> Unit {
  if self.phase == LUNA_CONTROL_SERVER_CLOSED || self.drain_requested {
    return
  }
  // Readiness becomes false before the listener is closed on the reactor.
  self.drain_requested = true
}

///|
fn LunaOnlineHttpControlServer::progress_drain(
  self : LunaOnlineHttpControlServer,
) -> LunaOnlineHttpControlProgress {
  if self.listeners.length() == 1 {
    self.listeners[0].close()
    self.listeners.clear()
    self.phase = LUNA_CONTROL_SERVER_DRAINING
    return LunaOnlineHttpControlDrainingProgress
  }
  if self.connections.length() == 1 {
    self.retire_connection()
    return LunaOnlineHttpControlDrainingProgress
  }
  self.phase = LUNA_CONTROL_SERVER_CLOSED
  LunaOnlineHttpControlClosedProgress
}
