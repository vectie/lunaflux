///|
fn LunaOnlineFramedConnectionPool::capture_observation(
  self : LunaOnlineFramedConnectionPool,
) -> LunaOnlineFramedConnectionPoolProgress {
  let observation = self.stream().take_observation() catch {
      _ => {
        self.fail_and_drain()
        return LunaOnlineFramedConnectionPoolFailedProgress
      }
    }
  if self.telemetry.record_observation(observation) < 0 {
    self.fail_and_drain()
    return LunaOnlineFramedConnectionPoolFailedProgress
  }
  observation.ack() catch {
    _ => {
      self.fail_and_drain()
      return LunaOnlineFramedConnectionPoolFailedProgress
    }
  }
  LunaOnlineFramedConnectionPoolAdvanced
}

///|
fn luna_online_framed_pool_rejection_reason(
  rule : @online_session.LunaOnlineFramedRejectionRule,
) -> @instance_log.LunaInstanceLogReason {
  match rule {
    LunaOnlineFramedRejectedDeadline => LunaLogDeadline
    LunaOnlineFramedRejectedCapacity => LunaLogCapacity
    LunaOnlineFramedRejectedService => LunaLogInternal
    _ => LunaLogValidation
  }
}

///|
fn LunaOnlineFramedConnectionPool::capture_rejection(
  self : LunaOnlineFramedConnectionPool,
) -> LunaOnlineFramedConnectionPoolProgress {
  let rejection = self.stream().take_rejection() catch {
      _ => {
        self.fail_and_drain()
        return LunaOnlineFramedConnectionPoolFailedProgress
      }
    }
  let route = rejection.request_route() catch {
    _ => {
      rejection.ack() catch {
        _ => ()
      }
      self.fail_and_drain()
      return LunaOnlineFramedConnectionPoolFailedProgress
    }
  }
  let index = self.find_route_slot(route)
  let rule = rejection.rule() catch {
    _ => {
      rejection.ack() catch {
        _ => ()
      }
      self.fail_and_drain()
      return LunaOnlineFramedConnectionPoolFailedProgress
    }
  }
  rejection.ack() catch {
    _ => {
      self.fail_and_drain()
      return LunaOnlineFramedConnectionPoolFailedProgress
    }
  }
  if index < 0 {
    self.fail_and_drain()
    return LunaOnlineFramedConnectionPoolFailedProgress
  }
  self.telemetry.record_network_rejection(
    luna_online_framed_pool_rejection_reason(rule),
  )
  self.begin_slot_retirement(index)
  LunaOnlineFramedConnectionPoolAdvanced
}

///|
fn LunaOnlineFramedConnectionPool::progress_service_once(
  self : LunaOnlineFramedConnectionPool,
) -> LunaOnlineFramedConnectionPoolProgress {
  if self.streams.length() != 1 || !self.stream().is_live() {
    return LunaOnlineFramedConnectionPoolBackpressured
  }
  let progress = self.stream().progress() catch {
      _ => {
        self.fail_and_drain()
        return LunaOnlineFramedConnectionPoolFailedProgress
      }
    }
  self.service_waiting = progress == LunaOnlineFramedCoordinatorWaiting
  match progress {
    LunaOnlineFramedCoordinatorObservationReady => self.capture_observation()
    LunaOnlineFramedCoordinatorRejectionReady => self.capture_rejection()
    LunaOnlineFramedCoordinatorSemanticEventReady =>
      self.capture_semantic_event()
    LunaOnlineFramedCoordinatorDraining | LunaOnlineFramedCoordinatorClosed => {
      self.fail_and_drain()
      LunaOnlineFramedConnectionPoolFailedProgress
    }
    LunaOnlineFramedCoordinatorIdle
    | LunaOnlineFramedCoordinatorWaiting
    | LunaOnlineFramedCoordinatorAwaitingInput =>
      LunaOnlineFramedConnectionPoolBackpressured
    _ => LunaOnlineFramedConnectionPoolAdvanced
  }
}

///|
fn LunaOnlineFramedConnectionPool::progress_slot_local(
  self : LunaOnlineFramedConnectionPool,
  index : Int,
) -> LunaOnlineFramedConnectionPoolProgress {
  let slot = self.slots[index]
  match slot.phase {
    1 => self.offer_slot_request_tail(index)
    2 => self.progress_slot_validation(index)
    3 => self.progress_slot_feed(index)
    5 => self.progress_slot_encoding(index)
    6 => self.progress_slot_event_copy(index)
    8 =>
      if self.try_finish_slot_retirement(index) {
        LunaOnlineFramedConnectionRetired
      } else {
        LunaOnlineFramedConnectionPoolBackpressured
      }
    _ => LunaOnlineFramedConnectionPoolBackpressured
  }
}

///|
fn LunaOnlineFramedConnectionPool::progress_one_local(
  self : LunaOnlineFramedConnectionPool,
) -> LunaOnlineFramedConnectionPoolProgress? {
  let start = self.io_cursor
  let mut saw_backpressure = false
  for offset in 0..<self.slots.length() {
    let index = (start + offset) % self.slots.length()
    let slot = self.slots[index]
    let has_local_progress = match slot.phase {
      LUNA_FRAMED_POOL_RECEIVING => slot.input_tail_length > 0
      LUNA_FRAMED_POOL_VALIDATING
      | LUNA_FRAMED_POOL_FEEDING
      | LUNA_FRAMED_POOL_ENCODING
      | LUNA_FRAMED_POOL_COPYING
      | LUNA_FRAMED_POOL_RETIRING => true
      _ => false
    }
    if has_local_progress {
      self.io_cursor = (index + 1) % self.slots.length()
      let progress = self.progress_slot_local(index)
      if progress != LunaOnlineFramedConnectionPoolBackpressured {
        return Some(progress)
      }
      saw_backpressure = true
    }
  }
  if saw_backpressure {
    Some(LunaOnlineFramedConnectionPoolBackpressured)
  } else {
    None
  }
}

///|
fn luna_online_framed_local_progress_preempts_writer(
  progress : LunaOnlineFramedConnectionPoolProgress,
) -> Bool {
  progress != LunaOnlineFramedConnectionPoolBackpressured
}

///|
async fn LunaOnlineFramedConnectionPool::progress_one_writer(
  self : LunaOnlineFramedConnectionPool,
) -> LunaOnlineFramedConnectionPoolProgress? {
  for offset in 0..<self.slots.length() {
    let index = (self.io_cursor + offset) % self.slots.length()
    if self.slots[index].phase == LUNA_FRAMED_POOL_WRITING {
      self.io_cursor = (index + 1) % self.slots.length()
      return Some(self.write_slot_output(index))
    }
  }
  None
}

///|
async fn LunaOnlineFramedConnectionPool::progress_one_input_io(
  self : LunaOnlineFramedConnectionPool,
  poll_timeout_millis : Int,
) -> LunaOnlineFramedConnectionPoolProgress? {
  for offset in 0..<self.slots.length() {
    let index = (self.io_cursor + offset) % self.slots.length()
    if self.slots[index].phase == LUNA_FRAMED_POOL_RECEIVING {
      self.io_cursor = (index + 1) % self.slots.length()
      return Some(self.read_slot_request(index, poll_timeout_millis))
    }
  }
  None
}

///|
fn LunaOnlineFramedConnectionPool::progress_drain(
  self : LunaOnlineFramedConnectionPool,
) -> LunaOnlineFramedConnectionPoolProgress {
  if self.active_connections > 0 {
    match self.progress_one_local() {
      Some(_) => ()
      None => ()
    }
    ignore(self.progress_service_once())
    return if self.failed {
      LunaOnlineFramedConnectionPoolFailedProgress
    } else {
      LunaOnlineFramedConnectionPoolDrainingProgress
    }
  }
  if self.streams.length() == 1 && self.stream().is_live() {
    if !self.stream_disconnect_started {
      self.stream().disconnect() catch {
        _ => {
          self.failed = true
          return LunaOnlineFramedConnectionPoolFailedProgress
        }
      }
      self.stream_disconnect_started = true
      return LunaOnlineFramedConnectionPoolDrainingProgress
    }
    ignore(
      self.stream().progress() catch {
        _ => {
          self.failed = true
          return LunaOnlineFramedConnectionPoolFailedProgress
        }
      },
    )
    return LunaOnlineFramedConnectionPoolDrainingProgress
  }
  if !self.service_drain_started {
    self.streams.clear()
    self.service().begin_drain()
    self.service_drain_started = true
  }
  let readiness = self.service().progress() catch {
      _ => {
        self.failed = true
        return LunaOnlineFramedConnectionPoolFailedProgress
      }
    }
  if readiness == LunaOnlineFramedServiceClosed {
    self.services.clear()
    self.closed = true
    LunaOnlineFramedConnectionPoolClosedProgress
  } else if self.failed {
    LunaOnlineFramedConnectionPoolFailedProgress
  } else {
    LunaOnlineFramedConnectionPoolDrainingProgress
  }
}

///|
/// Advances local and service work before polling an idle socket.
///
/// A live inference request crosses many cooperative service states. Polling
/// every active peer before each such state adds the socket timeout to every
/// token-step transition. Writable output remains first so an event credit is
/// released promptly; otherwise service work is drained until it reports real
/// backpressure, and only then may one input/listener poll block this turn.
pub async fn LunaOnlineFramedConnectionPool::progress_on_reactor(
  self : LunaOnlineFramedConnectionPool,
) -> LunaOnlineFramedConnectionPoolProgress {
  if self.closed {
    return LunaOnlineFramedConnectionPoolClosedProgress
  }
  if self.drain_requested {
    return self.progress_drain()
  }
  match self.progress_one_local() {
    Some(progress) =>
      if luna_online_framed_local_progress_preempts_writer(progress) {
        return progress
      }
    // A retiring slot may still be waiting for lower route retirement while
    // another slot owns a fully encoded event. Do not let that local
    // backpressure bypass the writer: advancing the service first could
    // publish the next event for a slot whose previous event is still in
    // flight, forcing a spurious transport cancellation.
    None => ()
  }
  match self.progress_one_writer() {
    Some(progress) => return progress
    None => ()
  }
  let service_progress = self.progress_service_once()
  if service_progress != LunaOnlineFramedConnectionPoolBackpressured {
    return service_progress
  }
  if self.service_waiting && self.service_probe_cooldown > 0 {
    self.service_probe_cooldown -= 1
    return service_progress
  }
  let can_accept = self.active_connections < self.slots.length()
  let poll_timeout_millis = if self.service_waiting {
    0
  } else {
    self.pool_limits.io_poll_timeout_millis
  }
  if can_accept && (self.service_waiting || self.accept_turn) {
    self.accept_turn = false
    let accepted = self.accept_one(poll_timeout_millis)
    if accepted != LunaOnlineFramedConnectionPoolBackpressured {
      self.service_probe_cooldown = 0
      return accepted
    }
  }
  self.accept_turn = can_accept
  // Active one-request connections are observed by response writes and the
  // request deadline. Speculative reads race with an idle peer and add a
  // socket poll to the token-step reactor.
  match self.progress_one_input_io(poll_timeout_millis) {
    Some(progress) => {
      self.service_probe_cooldown = if self.service_waiting &&
        progress == LunaOnlineFramedConnectionPoolBackpressured {
        32
      } else {
        0
      }
      progress
    }
    None => {
      if self.service_waiting {
        self.service_probe_cooldown = 32
      }
      service_progress
    }
  }
}
