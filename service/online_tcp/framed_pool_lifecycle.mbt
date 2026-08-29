///|
fn LunaOnlineFramedConnectionPool::service(
  self : LunaOnlineFramedConnectionPool,
) -> @online_session.LunaOnlineFramedService {
  if self.services.length() != 1 {
    abort("Luna framed pool service invariant")
  }
  self.services[0]
}

///|
fn LunaOnlineFramedConnectionPool::stream(
  self : LunaOnlineFramedConnectionPool,
) -> @online_session.LunaOnlineFramedStream {
  if self.streams.length() != 1 {
    abort("Luna framed pool stream invariant")
  }
  self.streams[0]
}

///|
fn LunaOnlineFramedConnectionSlot::route(
  self : LunaOnlineFramedConnectionSlot,
) -> @online_session.LunaOnlineFramedRequestRoute {
  if self.routes.length() != 1 {
    abort("Luna framed pool route invariant")
  }
  self.routes[0]
}

///|
fn LunaOnlineFramedConnectionSlot::receipt_remaining(
  self : LunaOnlineFramedConnectionSlot,
) -> Int {
  if self.receipts.length() != 1 {
    return 0
  }
  self.receipts[0].remaining_millis() catch {
    _ => 0
  }
}

///|
fn LunaOnlineFramedConnectionSlot::abort_local_authorities(
  self : LunaOnlineFramedConnectionSlot,
) -> Unit {
  if self.receipts.length() == 1 {
    self.receipts[0].abort() catch {
      _ => ()
    }
    self.receipts.clear()
  }
  if self.request_works.length() == 1 {
    self.request_works[0].abort() catch {
      _ => ()
    }
    self.request_works.clear()
  }
  if self.request_views.length() == 1 {
    self.request_views[0].release() catch {
      _ => ()
    }
    self.request_views.clear()
  }
  if self.event_works.length() == 1 {
    self.event_works[0].abort() catch {
      _ => ()
    }
    self.event_works.clear()
  }
  if self.event_views.length() == 1 {
    self.event_views[0].release() catch {
      _ => ()
    }
    self.event_views.clear()
  }
  if self.semantic_events.length() == 1 {
    if self.cancel_requested {
      self.semantic_events[0].discard_after_transport_cut() catch {
        _ => ()
      }
    } else {
      self.semantic_events[0].cancel_and_discard() catch {
        _ => ()
      }
      self.cancel_requested = true
    }
    self.semantic_events.clear()
  }
  if self.output_writes.length() == 1 {
    self.output_writes[0].abort() catch {
      _ => ()
    }
    self.output_writes.clear()
  }
  if self.flights.length() == 1 {
    self.flights[0].release() catch {
      _ => ()
    }
    self.flights.clear()
  }
}

///|
fn LunaOnlineFramedConnectionPool::close_slot_transport(
  self : LunaOnlineFramedConnectionPool,
  index : Int,
) -> Unit {
  let slot = self.slots[index]
  if slot.connections.length() == 1 {
    slot.connections[0].close()
    slot.connections.clear()
    self.network_disconnects += 1UL
    self.telemetry.record_network_disconnect(LunaLogPeerClosed)
  }
}

///|
fn LunaOnlineFramedConnectionPool::begin_slot_retirement(
  self : LunaOnlineFramedConnectionPool,
  index : Int,
) -> Unit {
  let slot = self.slots[index]
  if slot.phase == LUNA_FRAMED_POOL_FREE ||
    slot.phase == LUNA_FRAMED_POOL_RETIRING {
    return
  }
  self.close_slot_transport(index)
  slot.abort_local_authorities()
  if self.ingress_slot == index {
    self.ingress_slot = -1
  }
  if slot.routes.length() == 1 &&
    !slot.terminal_delivered &&
    !slot.cancel_requested {
    self.stream().request_cancel_route(slot.route()) catch {
      _ => ()
    }
    slot.cancel_requested = true
  }
  slot.phase = LUNA_FRAMED_POOL_RETIRING
}

///|
fn LunaOnlineFramedConnectionSlot::reset_for_reuse(
  self : LunaOnlineFramedConnectionSlot,
) -> Unit {
  if !self.connections.is_empty() ||
    !self.receipts.is_empty() ||
    !self.request_works.is_empty() ||
    !self.request_views.is_empty() ||
    !self.event_works.is_empty() ||
    !self.event_views.is_empty() ||
    !self.semantic_events.is_empty() ||
    !self.output_writes.is_empty() ||
    !self.flights.is_empty() ||
    !self.output.is_idle() {
    abort("Luna framed pool slot reuse invariant")
  }
  self.routes.clear()
  self.input_cursor = 0
  self.input_tail_start = 0
  self.input_tail_length = 0
  self.feed_cursor = 0
  self.event_copy_cursor = 0
  self.output_length = 0
  self.output_cursor = 0
  self.event_terminal = false
  self.terminal_delivered = false
  self.cancel_requested = false
  self.phase = LUNA_FRAMED_POOL_FREE
}

///|
fn LunaOnlineFramedConnectionPool::try_finish_slot_retirement(
  self : LunaOnlineFramedConnectionPool,
  index : Int,
) -> Bool {
  let slot = self.slots[index]
  if slot.phase != LUNA_FRAMED_POOL_RETIRING || !slot.semantic_events.is_empty() {
    return false
  }
  let route_live = if slot.routes.length() == 1 {
    self.stream().request_route_live(slot.route()) catch {
      _ => true
    }
  } else {
    false
  }
  if route_live {
    return false
  }
  slot.reset_for_reuse()
  self.active_connections -= 1
  true
}

///|
fn LunaOnlineFramedConnectionPool::find_route_slot(
  self : LunaOnlineFramedConnectionPool,
  route : @online_session.LunaOnlineFramedRequestRoute,
) -> Int {
  for index, slot in self.slots {
    if slot.routes.length() == 1 && slot.route().same_request(route) {
      return index
    }
  }
  -1
}

///|
pub fn LunaOnlineFramedConnectionPool::local_addr(
  self : LunaOnlineFramedConnectionPool,
) -> @socket.Addr {
  self.bound_addr
}

///|
pub fn LunaOnlineFramedConnectionPool::active_connection_count(
  self : LunaOnlineFramedConnectionPool,
) -> Int {
  self.active_connections
}

///|
pub fn LunaOnlineFramedConnectionPool::network_accepts(
  self : LunaOnlineFramedConnectionPool,
) -> UInt64 {
  self.network_accepts
}

///|
pub fn LunaOnlineFramedConnectionPool::network_disconnects(
  self : LunaOnlineFramedConnectionPool,
) -> UInt64 {
  self.network_disconnects
}

///|
pub fn LunaOnlineFramedConnectionPool::delivered_events(
  self : LunaOnlineFramedConnectionPool,
) -> UInt64 {
  self.delivered_events
}

///|
pub fn LunaOnlineFramedConnectionPool::state(
  self : LunaOnlineFramedConnectionPool,
) -> LunaOnlineFramedConnectionPoolState {
  if self.closed {
    LunaOnlineFramedConnectionPoolClosed
  } else if self.failed {
    LunaOnlineFramedConnectionPoolFailed
  } else if self.drain_requested {
    LunaOnlineFramedConnectionPoolDraining
  } else if self.active_connections > 0 {
    LunaOnlineFramedConnectionPoolActive
  } else {
    LunaOnlineFramedConnectionPoolListening
  }
}

///|
pub fn LunaOnlineFramedConnectionPool::readiness(
  self : LunaOnlineFramedConnectionPool,
) -> LunaOnlineFramedConnectionPoolReadiness {
  if !self.closed &&
    !self.failed &&
    !self.drain_requested &&
    self.listeners.length() == 1 &&
    self.services.length() == 1 &&
    self.streams.length() == 1 &&
    self.stream().is_live() {
    LunaOnlineFramedConnectionPoolReady
  } else {
    LunaOnlineFramedConnectionPoolNotReady
  }
}

///|
pub fn LunaOnlineFramedConnectionPool::begin_drain(
  self : LunaOnlineFramedConnectionPool,
) -> Unit {
  if self.closed || self.drain_requested {
    return
  }
  self.drain_requested = true
  if self.listeners.length() == 1 {
    self.listeners[0].close()
    self.listeners.clear()
  }
  for index, slot in self.slots {
    if slot.phase != LUNA_FRAMED_POOL_FREE {
      self.begin_slot_retirement(index)
    }
  }
}

///|
pub fn LunaOnlineFramedConnectionPool::health(
  self : LunaOnlineFramedConnectionPool,
) -> LunaOnlineTcpServerHealth {
  if self.state() == LunaOnlineFramedConnectionPoolFailed ||
    self.state() == LunaOnlineFramedConnectionPoolClosed {
    LunaOnlineTcpServerFailed
  } else {
    LunaOnlineTcpServerHealthy
  }
}

///|
fn LunaOnlineFramedConnectionPool::fail_and_drain(
  self : LunaOnlineFramedConnectionPool,
) -> Unit {
  self.failed = true
  self.begin_drain()
}
