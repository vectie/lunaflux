///|
fn luna_online_tcp_rule_code(rule : LunaOnlineTcpRule) -> Int {
  match rule {
    LunaOnlineTcpLimits => 1
    LunaOnlineTcpPreparation => 2
    LunaOnlineTcpBind => 3
    LunaOnlineTcpAccept => 4
    LunaOnlineTcpRead => 5
    LunaOnlineTcpWrite => 6
    LunaOnlineTcpTimeout => 7
    LunaOnlineTcpCancelled => 8
    LunaOnlineTcpCoordinator => 9
    LunaOnlineTcpLifecycle => 10
    LunaOnlineTcpOwnership => 11
    LunaOnlineTcpTelemetry => 12
  }
}

///|
fn luna_online_tcp_rule_from_code(code : Int) -> LunaOnlineTcpRule {
  match code {
    1 => LunaOnlineTcpLimits
    2 => LunaOnlineTcpPreparation
    3 => LunaOnlineTcpBind
    4 => LunaOnlineTcpAccept
    5 => LunaOnlineTcpRead
    6 => LunaOnlineTcpWrite
    7 => LunaOnlineTcpTimeout
    8 => LunaOnlineTcpCancelled
    9 => LunaOnlineTcpCoordinator
    10 => LunaOnlineTcpLifecycle
    11 => LunaOnlineTcpOwnership
    12 => LunaOnlineTcpTelemetry
    _ => abort("Luna online TCP failure rule invariant")
  }
}

///|
fn LunaOnlineTcpEndpoint::coordinator(
  self : LunaOnlineTcpEndpoint,
) -> @online_session.LunaOnlineFramedCoordinator {
  if self.coordinators.length() != 1 {
    abort("Luna online TCP coordinator invariant")
  }
  self.coordinators[0]
}

///|
fn LunaOnlineTcpEndpoint::latch_failure(
  self : LunaOnlineTcpEndpoint,
  rule : LunaOnlineTcpRule,
) -> Unit {
  if self.failure_rule == 0 {
    self.failure_rule = luna_online_tcp_rule_code(rule)
  }
}

///|
fn LunaOnlineTcpEndpoint::release_revoked_output(
  self : LunaOnlineTcpEndpoint,
) -> Unit {
  self.offers.clear()
  if self.flights.length() == 1 {
    try! self.flights[0].release()
    self.flights.clear()
  }
}

///|
fn LunaOnlineTcpEndpoint::begin_disconnect(
  self : LunaOnlineTcpEndpoint,
  failure? : LunaOnlineTcpRule,
) -> Unit {
  if failure is Some(rule) {
    self.latch_failure(rule)
  }
  self.disconnect_requested = true
  self.drain_requested = true
  self.tail_start = 0
  self.tail_length = 0
  self.tail_backpressured = false
  self.close_pending = self.servers.length() == 1 ||
    self.connections.length() == 1
  let revoked = try {
    self.coordinator().disconnect()
    true
  } catch {
    _ => false
  }
  if revoked {
    self.release_revoked_output()
  } else {
    self.latch_failure(LunaOnlineTcpCoordinator)
  }
  self.phase = if self.close_pending {
    LUNA_ONLINE_TCP_CLOSE_REQUIRED
  } else {
    LUNA_ONLINE_TCP_DRAINING
  }
}

///|
fn LunaOnlineTcpEndpoint::close_transport_on_reactor(
  self : LunaOnlineTcpEndpoint,
) -> Unit {
  if self.connections.length() == 1 {
    self.connections[0].close()
    self.connections.clear()
  }
  if self.servers.length() == 1 {
    self.servers[0].close()
    self.servers.clear()
  }
  self.close_pending = false
  if self.phase != LUNA_ONLINE_TCP_CLOSED {
    self.phase = LUNA_ONLINE_TCP_DRAINING
  }
}

///|
fn LunaOnlineTcpEndpoint::publish_closed_if_ready(
  self : LunaOnlineTcpEndpoint,
) -> Bool {
  if self.coordinator().is_closed() &&
    self.servers.is_empty() &&
    self.connections.is_empty() {
    self.phase = LUNA_ONLINE_TCP_CLOSED
    true
  } else {
    false
  }
}

///|
fn LunaOnlineTcpEndpoint::bounded_transport_wait(
  self : LunaOnlineTcpEndpoint,
  configured_millis : Int,
) -> Int raise LunaOnlineTcpError {
  let remaining = self.coordinator().transport_wait_remaining_millis() catch {
      _ => raise luna_online_tcp_error(LunaOnlineTcpCoordinator)
    }
  // Leave one millisecond for timer installation and the coordinator's own
  // post-I/O hard-deadline recheck. A shorter obligation must not start I/O.
  luna_online_tcp_safe_wait(configured_millis, remaining)
}

///|
pub fn LunaOnlineTcpEndpoint::local_addr(
  self : LunaOnlineTcpEndpoint,
) -> @socket.Addr {
  self.bound_addr
}

///|
pub fn LunaOnlineTcpEndpoint::state(
  self : LunaOnlineTcpEndpoint,
) -> LunaOnlineTcpEndpointState {
  match self.phase {
    0 => LunaOnlineTcpListening
    1 => LunaOnlineTcpAccepting
    2 => LunaOnlineTcpConnected
    3 => LunaOnlineTcpDraining
    4 => LunaOnlineTcpCloseRequired
    5 => LunaOnlineTcpClosed
    _ => abort("Luna online TCP endpoint phase invariant")
  }
}

///|
pub fn LunaOnlineTcpEndpoint::failure_rule(
  self : LunaOnlineTcpEndpoint,
) -> LunaOnlineTcpRule raise LunaOnlineTcpError {
  if self.failure_rule == 0 {
    raise luna_online_tcp_error(LunaOnlineTcpLifecycle)
  }
  luna_online_tcp_rule_from_code(self.failure_rule)
}

///|
fn luna_online_tcp_rejection_code(
  rule : @online_session.LunaOnlineFramedRejectionRule,
) -> Int {
  match rule {
    LunaOnlineFramedRejectedFrame => 1
    LunaOnlineFramedRejectedIdentity => 2
    LunaOnlineFramedRejectedDeadline => 3
    LunaOnlineFramedRejectedInput => 4
    LunaOnlineFramedRejectedTokenization => 5
    LunaOnlineFramedRejectedCapacity => 6
    LunaOnlineFramedRejectedService => 7
  }
}

///|
fn luna_online_tcp_rejection_from_code(
  code : Int,
) -> LunaOnlineTcpRejectionRule {
  match code {
    1 => LunaOnlineTcpRejectedFrame
    2 => LunaOnlineTcpRejectedIdentity
    3 => LunaOnlineTcpRejectedDeadline
    4 => LunaOnlineTcpRejectedInput
    5 => LunaOnlineTcpRejectedTokenization
    6 => LunaOnlineTcpRejectedCapacity
    7 => LunaOnlineTcpRejectedService
    _ => abort("Luna online TCP rejection rule invariant")
  }
}

///|
pub fn LunaOnlineTcpEndpoint::rejection_sequence(
  self : LunaOnlineTcpEndpoint,
) -> UInt64 raise LunaOnlineTcpError {
  if !self.rejection_latched {
    raise luna_online_tcp_error(LunaOnlineTcpLifecycle)
  }
  self.rejection_sequence
}

///|
pub fn LunaOnlineTcpEndpoint::rejection_rule(
  self : LunaOnlineTcpEndpoint,
) -> LunaOnlineTcpRejectionRule raise LunaOnlineTcpError {
  if !self.rejection_latched {
    raise luna_online_tcp_error(LunaOnlineTcpLifecycle)
  }
  luna_online_tcp_rejection_from_code(self.rejection_rule)
}

///|
pub fn LunaOnlineTcpEndpoint::begin_drain(
  self : LunaOnlineTcpEndpoint,
) -> Unit raise LunaOnlineTcpError {
  if self.activity.reactor_active || self.activity.maintenance_active {
    raise luna_online_tcp_error(LunaOnlineTcpOwnership)
  }
  if self.phase == LUNA_ONLINE_TCP_CLOSED {
    return
  }
  self.drain_requested = true
  self.tail_start = 0
  self.tail_length = 0
  self.tail_backpressured = false
  self.coordinator().begin_drain()
  if self.connections.is_empty() {
    self.close_pending = self.servers.length() == 1
    self.phase = if self.close_pending {
      LUNA_ONLINE_TCP_CLOSE_REQUIRED
    } else {
      LUNA_ONLINE_TCP_DRAINING
    }
  }
}

///|
pub fn LunaOnlineTcpEndpoint::disconnect(
  self : LunaOnlineTcpEndpoint,
) -> Unit raise LunaOnlineTcpError {
  if self.activity.reactor_active || self.activity.maintenance_active {
    raise luna_online_tcp_error(LunaOnlineTcpOwnership)
  }
  if self.phase != LUNA_ONLINE_TCP_CLOSED {
    self.begin_disconnect()
  }
}

///|
pub fn LunaOnlineTcpEndpoint::request_cancel(
  self : LunaOnlineTcpEndpoint,
) -> Unit raise LunaOnlineTcpError {
  if self.activity.reactor_active || self.activity.maintenance_active {
    raise luna_online_tcp_error(LunaOnlineTcpOwnership)
  }
  self.coordinator().request_cancel() catch {
    _ => raise luna_online_tcp_error(LunaOnlineTcpCoordinator)
  }
}
