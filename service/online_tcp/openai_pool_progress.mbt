///|
fn LunaOnlineOpenAIConnectionPool::capture_observation(
  self : LunaOnlineOpenAIConnectionPool,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  let observation = self.stream().take_observation() catch {
      _ => {
        self.begin_drain()
        return LunaOnlineOpenAIConnectionPoolDrainingProgress
      }
    }
  if self.telemetry.record_observation(observation) < 0 {
    self.begin_drain()
    return LunaOnlineOpenAIConnectionPoolDrainingProgress
  }
  observation.ack() catch {
    _ => self.begin_drain()
  }
  LunaOnlineOpenAIConnectionPoolAdvanced
}

///|
fn LunaOnlineOpenAIConnectionPool::capture_rejection(
  self : LunaOnlineOpenAIConnectionPool,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  let rejection = self.stream().take_rejection() catch {
      _ => {
        self.begin_drain()
        return LunaOnlineOpenAIConnectionPoolDrainingProgress
      }
    }
  let route = rejection.request_route() catch {
    _ => {
      self.begin_drain()
      return LunaOnlineOpenAIConnectionPoolDrainingProgress
    }
  }
  let index = self.find_route_slot(route)
  if index < 0 {
    rejection.ack() catch {
      _ => self.begin_drain()
    }
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  let slot = self.slots[index]
  let rejection_rule = rejection.rule() catch {
    _ => {
      self.begin_drain()
      return LunaOnlineOpenAIConnectionPoolDrainingProgress
    }
  }
  let kind : @http1.LunaHttp1ErrorResponseKind = match rejection_rule {
    LunaOnlineFramedRejectedCapacity => LunaHttp1CapacityError
    LunaOnlineFramedRejectedService | LunaOnlineFramedRejectedDeadline =>
      LunaHttp1ServiceError
    _ => LunaHttp1ValidationError
  }
  if slot.sse_started ||
    slot.phase == LUNA_OPENAI_POOL_RETIRING ||
    !slot.output.is_idle() {
    rejection.ack() catch {
      _ => self.begin_drain()
    }
    slot.cancel_requested = true
    self.stream().request_cancel_route(route) catch {
      _ => ()
    }
    self.begin_slot_retirement(index)
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  if !slot.begin_http_error(kind) {
    rejection.ack() catch {
      _ => self.begin_drain()
    }
    self.begin_slot_retirement(index)
    return LunaOnlineOpenAIConnectionPoolAdvanced
  }
  slot.rejections.push(rejection)
  LunaOnlineOpenAIConnectionPoolAdvanced
}

///|
fn LunaOnlineOpenAIConnectionPool::progress_service_once(
  self : LunaOnlineOpenAIConnectionPool,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  if self.streams.length() != 1 || !self.stream().is_live() {
    return LunaOnlineOpenAIConnectionPoolBackpressured
  }
  let progress = self.stream().progress() catch {
      _ => {
        self.begin_drain()
        return LunaOnlineOpenAIConnectionPoolDrainingProgress
      }
    }
  match progress {
    LunaOnlineFramedCoordinatorObservationReady => self.capture_observation()
    LunaOnlineFramedCoordinatorRejectionReady => self.capture_rejection()
    LunaOnlineFramedCoordinatorSemanticEventReady =>
      self.capture_semantic_event()
    LunaOnlineFramedCoordinatorDraining | LunaOnlineFramedCoordinatorClosed => {
      self.begin_drain()
      LunaOnlineOpenAIConnectionPoolDrainingProgress
    }
    LunaOnlineFramedCoordinatorIdle
    | LunaOnlineFramedCoordinatorAwaitingInput =>
      LunaOnlineOpenAIConnectionPoolBackpressured
    _ => LunaOnlineOpenAIConnectionPoolAdvanced
  }
}

///|
async fn LunaOnlineOpenAIConnectionPool::progress_slot(
  self : LunaOnlineOpenAIConnectionPool,
  index : Int,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  let slot = self.slots[index]
  match slot.phase {
    0 => LunaOnlineOpenAIConnectionPoolBackpressured
    5 =>
      if slot.terminal_observed {
        self.begin_slot_retirement(index)
        LunaOnlineOpenAIConnectionPoolAdvanced
      } else {
        LunaOnlineOpenAIConnectionPoolBackpressured
      }
    1 =>
      if slot.tail_length > 0 {
        self.offer_slot_http_tail(index)
      } else {
        self.read_slot_http(index)
      }
    2 => self.progress_slot_http(index)
    3 => self.progress_slot_inbound(index)
    4 => self.progress_slot_handoff(index)
    6 => self.progress_slot_compat(index)
    7 => self.progress_slot_response(index)
    8 => self.write_slot_output(index)
    9 =>
      if self.try_finish_slot_retirement(index) {
        LunaOnlineOpenAIConnectionRetired
      } else {
        LunaOnlineOpenAIConnectionPoolBackpressured
      }
    10 => self.write_slot_embedded_control(index)
    _ => {
      self.begin_slot_retirement(index)
      LunaOnlineOpenAIConnectionPoolAdvanced
    }
  }
}

///|
async fn LunaOnlineOpenAIConnectionPool::progress_one_slot(
  self : LunaOnlineOpenAIConnectionPool,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  // Progress retained parser, renderer, semantic-credit, output, and
  // retirement work before polling an idle input socket. Otherwise one idle
  // client can be selected first and the shared stream can be re-entered while
  // another slot still owns its previous semantic event.
  for offset in 0..<self.slots.length() {
    let index = (self.io_cursor + offset) % self.slots.length()
    let slot = self.slots[index]
    let local_progress = match slot.phase {
      LUNA_OPENAI_POOL_HTTP_PROGRESS
      | LUNA_OPENAI_POOL_INBOUND_PROGRESS
      | LUNA_OPENAI_POOL_HANDOFF_FEED
      | LUNA_OPENAI_POOL_COMPAT_PROGRESS
      | LUNA_OPENAI_POOL_RESPONSE_PROGRESS
      | LUNA_OPENAI_POOL_OUTPUT_WRITE
      | LUNA_OPENAI_POOL_CONTROL_WRITE
      | LUNA_OPENAI_POOL_RETIRING => true
      LUNA_OPENAI_POOL_HTTP_RECEIVE => slot.tail_length > 0
      LUNA_OPENAI_POOL_REQUEST_ACTIVE => slot.terminal_observed
      _ => false
    }
    if local_progress {
      self.io_cursor = (index + 1) % self.slots.length()
      return self.progress_slot(index)
    }
  }
  // At most one idle-input socket poll is performed per owner transition.
  for offset in 0..<self.slots.length() {
    let index = (self.io_cursor + offset) % self.slots.length()
    if self.slots[index].phase == LUNA_OPENAI_POOL_HTTP_RECEIVE {
      self.io_cursor = (index + 1) % self.slots.length()
      return self.progress_slot(index)
    }
  }
  LunaOnlineOpenAIConnectionPoolBackpressured
}

///|
async fn LunaOnlineOpenAIConnectionPool::progress_drain(
  self : LunaOnlineOpenAIConnectionPool,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  if self.active_connections > 0 {
    ignore(self.progress_one_slot())
    ignore(self.progress_service_once())
    return LunaOnlineOpenAIConnectionPoolDrainingProgress
  }
  ignore(self.progress_service_once())
  if self.streams.length() == 1 && self.stream().is_live() {
    if !self.stream_disconnect_started {
      self.stream().disconnect() catch {
        _ => return LunaOnlineOpenAIConnectionPoolDrainingProgress
      }
      self.stream_disconnect_started = true
      return LunaOnlineOpenAIConnectionPoolDrainingProgress
    }
    ignore(
      self.stream().progress() catch {
        _ => return LunaOnlineOpenAIConnectionPoolDrainingProgress
      },
    )
    return LunaOnlineOpenAIConnectionPoolDrainingProgress
  }
  if !self.service_drain_started {
    self.streams.clear()
    self.service().begin_drain()
    self.service_drain_started = true
  }
  let readiness = self.service().progress() catch {
      _ => return LunaOnlineOpenAIConnectionPoolDrainingProgress
    }
  if readiness == LunaOnlineFramedServiceClosed {
    for slot in self.slots {
      slot.http_workspace.close_authentication()
    }
    self.auth_policy.close()
    self.closed = true
    LunaOnlineOpenAIConnectionPoolClosedProgress
  } else {
    LunaOnlineOpenAIConnectionPoolDrainingProgress
  }
}

///|
/// Advances at most one bounded socket transition before one shared service
/// transition. This ordering frees a completed output cell before another
/// semantic event may be emitted. Socket waits are short polls; no socket I/O
/// owns a semantic event credit or scheduler ticket.
pub async fn LunaOnlineOpenAIConnectionPool::progress_on_reactor(
  self : LunaOnlineOpenAIConnectionPool,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  if self.closed {
    return LunaOnlineOpenAIConnectionPoolClosedProgress
  }
  if self.drain_requested {
    return self.progress_drain()
  }
  let slot_progress = self.progress_one_slot()
  if self.drain_requested {
    return LunaOnlineOpenAIConnectionPoolDrainingProgress
  }
  // A local transition may have delivered an event into an immutable output
  // flight. Do not re-enter the shared service until a later reactor turn has
  // either written that whole cell or observed genuine socket backpressure.
  if slot_progress != LunaOnlineOpenAIConnectionPoolBackpressured {
    return slot_progress
  }
  let service_progress = self.progress_service_once()
  if self.drain_requested {
    return LunaOnlineOpenAIConnectionPoolDrainingProgress
  }
  let can_accept = self.active_connections < self.slots.length()
  if can_accept && self.accept_turn {
    self.accept_turn = false
    let accepted = self.accept_one()
    if accepted != LunaOnlineOpenAIConnectionPoolBackpressured {
      return accepted
    }
  }
  self.accept_turn = can_accept
  service_progress
}
