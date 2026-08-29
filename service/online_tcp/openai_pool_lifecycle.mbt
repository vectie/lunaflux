///|
fn LunaOnlineOpenAIConnectionPool::service(
  self : LunaOnlineOpenAIConnectionPool,
) -> @online_session.LunaOnlineFramedService {
  if self.services.length() != 1 {
    abort("Luna OpenAI pool service invariant")
  }
  self.services[0]
}

///|
fn LunaOnlineOpenAIConnectionPool::stream(
  self : LunaOnlineOpenAIConnectionPool,
) -> @online_session.LunaOnlineFramedStream {
  if self.streams.length() != 1 {
    abort("Luna OpenAI pool stream invariant")
  }
  self.streams[0]
}

///|
fn LunaOnlineOpenAIConnectionSlot::route(
  self : LunaOnlineOpenAIConnectionSlot,
) -> @online_session.LunaOnlineFramedRequestRoute {
  if self.routes.length() != 1 {
    abort("Luna OpenAI pool route invariant")
  }
  self.routes[0]
}

///|
fn LunaOnlineOpenAIConnectionSlot::abort_local_authorities(
  self : LunaOnlineOpenAIConnectionSlot,
) -> Unit {
  if self.receipts.length() == 1 {
    self.receipts[0].abort() catch {
      _ => ()
    }
    self.receipts.clear()
  }
  if self.http_works.length() == 1 {
    self.http_works[0].abort() catch {
      _ => ()
    }
    self.http_works.clear()
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
  if self.rejections.length() == 1 {
    self.rejections[0].ack() catch {
      _ => ()
    }
    self.rejections.clear()
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
  self.embedded_control.clear()
}

///|
fn LunaOnlineOpenAIConnectionPool::begin_slot_retirement(
  self : LunaOnlineOpenAIConnectionPool,
  index : Int,
) -> Unit {
  let slot = self.slots[index]
  if slot.phase == LUNA_OPENAI_POOL_FREE {
    return
  }
  if slot.connections.length() == 1 {
    slot.connections[0].close()
    slot.connections.clear()
    self.telemetry.record_network_disconnect(LunaLogPeerClosed)
  }
  slot.abort_local_authorities()
  if self.ingress_slot == index {
    self.ingress_slot = -1
  }
  if slot.routes.length() == 1 && !slot.cancel_requested {
    try self.stream().request_cancel_route(slot.route()) catch {
      _ => ()
    } noraise {
      _ => slot.cancel_requested = true
    }
  }
  slot.phase = LUNA_OPENAI_POOL_RETIRING
}

///|
fn LunaOnlineOpenAIConnectionSlot::reset_for_reuse(
  self : LunaOnlineOpenAIConnectionSlot,
) -> Unit {
  if !self.output.is_idle() ||
    !self.connections.is_empty() ||
    !self.receipts.is_empty() ||
    !self.http_works.is_empty() ||
    !self.response_works.is_empty() ||
    !self.response_views.is_empty() ||
    !self.inbound_works.is_empty() ||
    !self.inbound_views.is_empty() ||
    !self.outbound_works.is_empty() ||
    !self.outbound_views.is_empty() ||
    !self.semantic_events.is_empty() ||
    !self.rejections.is_empty() {
    abort("Luna OpenAI pool slot reuse invariant")
  }
  self.routes.clear()
  self.request_sequence = 0UL
  self.route_mode = 0
  self.tail_start = 0
  self.tail_length = 0
  self.response_kind = 0
  self.response_copy_cursor = 0
  self.compat_copy_cursor = 0
  self.output_length = 0
  self.output_cursor = 0
  self.event_terminal = false
  self.terminal_observed = false
  self.sse_started = false
  self.cancel_requested = false
  self.embedded_control.clear()
  self.phase = LUNA_OPENAI_POOL_FREE
}

///|
fn LunaOnlineOpenAIConnectionPool::try_finish_slot_retirement(
  self : LunaOnlineOpenAIConnectionPool,
  index : Int,
) -> Bool {
  let slot = self.slots[index]
  if slot.phase != LUNA_OPENAI_POOL_RETIRING ||
    !slot.semantic_events.is_empty() ||
    !slot.rejections.is_empty() {
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
fn LunaOnlineOpenAIConnectionPool::find_route_slot(
  self : LunaOnlineOpenAIConnectionPool,
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
pub fn LunaOnlineOpenAIConnectionPool::local_addr(
  self : LunaOnlineOpenAIConnectionPool,
) -> @socket.Addr {
  self.bound_addr
}

///|
pub fn LunaOnlineOpenAIConnectionPool::active_connection_count(
  self : LunaOnlineOpenAIConnectionPool,
) -> Int {
  self.active_connections
}

///|
pub fn LunaOnlineOpenAIConnectionPool::state(
  self : LunaOnlineOpenAIConnectionPool,
) -> LunaOnlineOpenAIConnectionPoolState {
  if self.closed {
    LunaOnlineOpenAIConnectionPoolClosed
  } else if self.drain_requested {
    LunaOnlineOpenAIConnectionPoolDraining
  } else if self.active_connections > 0 {
    LunaOnlineOpenAIConnectionPoolActive
  } else {
    LunaOnlineOpenAIConnectionPoolListening
  }
}

///|
pub fn LunaOnlineOpenAIConnectionPool::begin_drain(
  self : LunaOnlineOpenAIConnectionPool,
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
    if slot.phase != LUNA_OPENAI_POOL_FREE {
      self.begin_slot_retirement(index)
    }
  }
}

///|
pub fn LunaOnlineOpenAIConnectionPool::readiness(
  self : LunaOnlineOpenAIConnectionPool,
) -> LunaOnlineOpenAIConnectionPoolReadiness {
  if !self.closed &&
    !self.drain_requested &&
    self.listeners.length() == 1 &&
    self.services.length() == 1 &&
    self.streams.length() == 1 &&
    self.stream().is_live() {
    LunaOnlineOpenAIConnectionPoolReady
  } else {
    LunaOnlineOpenAIConnectionPoolNotReady
  }
}

///|
pub fn LunaOnlineOpenAIConnectionPool::health(
  self : LunaOnlineOpenAIConnectionPool,
) -> LunaOnlineTcpServerHealth {
  if self.closed {
    LunaOnlineTcpServerFailed
  } else {
    LunaOnlineTcpServerHealthy
  }
}
