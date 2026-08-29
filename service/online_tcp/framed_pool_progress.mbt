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
  for offset in 0..<self.slots.length() {
    let index = (self.io_cursor + offset) % self.slots.length()
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
      return Some(self.progress_slot_local(index))
    }
  }
  None
}

///|
async fn LunaOnlineFramedConnectionPool::progress_one_io(
  self : LunaOnlineFramedConnectionPool,
) -> LunaOnlineFramedConnectionPoolProgress? {
  // A published event cell must be released before the shared stream can
  // expose another event for that route. Drain every writable slot ahead of
  // idle input polls so a partial client cannot perturb event ownership.
  for offset in 0..<self.slots.length() {
    let index = (self.io_cursor + offset) % self.slots.length()
    if self.slots[index].phase == LUNA_FRAMED_POOL_WRITING {
      self.io_cursor = (index + 1) % self.slots.length()
      return Some(self.write_slot_output(index))
    }
  }
  for offset in 0..<self.slots.length() {
    let index = (self.io_cursor + offset) % self.slots.length()
    if self.slots[index].phase == LUNA_FRAMED_POOL_RECEIVING {
      self.io_cursor = (index + 1) % self.slots.length()
      return Some(self.read_slot_request(index))
    }
  }
  None
}

///|
fn LunaOnlineFramedConnectionPool::has_writing_slot(
  self : LunaOnlineFramedConnectionPool,
) -> Bool {
  self.slots.any(fn(slot) { slot.phase == LUNA_FRAMED_POOL_WRITING })
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
/// Advances bounded owner state and performs at most one short socket poll.
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
    Some(progress) => {
      if progress == LunaOnlineFramedConnectionPoolBackpressured {
        return self.progress_service_once()
      }
      return progress
    }
    None => ()
  }
  let can_accept = self.active_connections < self.slots.length()
  if can_accept && self.accept_turn {
    self.accept_turn = false
    return self.accept_one()
  }
  self.accept_turn = can_accept
  let polled_writer = self.has_writing_slot()
  match self.progress_one_io() {
    Some(progress) =>
      if progress == LunaOnlineFramedConnectionPoolBackpressured {
        if polled_writer {
          progress
        } else {
          self.progress_service_once()
        }
      } else {
        progress
      }
    None => self.progress_service_once()
  }
}
