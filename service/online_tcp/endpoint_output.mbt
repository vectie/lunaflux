///|
fn LunaOnlineTcpOutputWrite::copy_event(
  self : LunaOnlineTcpOutputWrite,
  coordinator : @online_session.LunaOnlineFramedCoordinator,
) -> @online_session.LunaOnlineFramedEventOffer raise {
  let scratch = self.require_mutating()
  coordinator.copy_framed_event_chunk_to(
    scratch.mutable,
    destination_offset=0,
    maximum_length=scratch.capacity,
  )
}

///|
async fn LunaOnlineTcpOutputFlight::write_once(
  self : LunaOnlineTcpOutputFlight,
  connection : @socket.Tcp,
) -> Int {
  let scratch = self.require_in_flight()
  connection.write_once(scratch.immutable, offset=0, len=self.length)
}

///|
async fn LunaOnlineTcpEndpoint::write_event_on_reactor(
  self : LunaOnlineTcpEndpoint,
) -> LunaOnlineTcpProgress {
  if self.connections.length() != 1 ||
    !self.offers.is_empty() ||
    !self.flights.is_empty() ||
    !self.output.is_idle() {
    self.begin_disconnect(failure=LunaOnlineTcpLifecycle)
    self.close_transport_on_reactor()
    return LunaOnlineTcpCleanupRequired
  }
  let write = self.output.begin_write() catch {
    _ => {
      self.begin_disconnect(failure=LunaOnlineTcpLifecycle)
      self.close_transport_on_reactor()
      return LunaOnlineTcpCleanupRequired
    }
  }
  let offer = write.copy_event(self.coordinator()) catch {
    _ => {
      try! write.abort()
      self.begin_disconnect(failure=LunaOnlineTcpCoordinator)
      self.close_transport_on_reactor()
      return LunaOnlineTcpCleanupRequired
    }
  }
  let length = offer.length() catch {
    _ => {
      try! write.abort()
      self.begin_disconnect(failure=LunaOnlineTcpCoordinator)
      self.close_transport_on_reactor()
      return LunaOnlineTcpCleanupRequired
    }
  }
  let flight = write.publish(length) catch {
    _ => {
      try! write.abort()
      self.begin_disconnect(failure=LunaOnlineTcpLifecycle)
      self.close_transport_on_reactor()
      return LunaOnlineTcpCleanupRequired
    }
  }
  self.offers.push(offer)
  self.flights.push(flight)
  let wait = self.bounded_transport_wait(self.limits.write_timeout_millis) catch {
    _ => {
      self.begin_disconnect(failure=LunaOnlineTcpCoordinator)
      self.close_transport_on_reactor()
      return LunaOnlineTcpCleanupRequired
    }
  }
  if wait <= 0 {
    self.begin_disconnect(failure=LunaOnlineTcpTimeout)
    self.close_transport_on_reactor()
    return LunaOnlineTcpCleanupRequired
  }
  let written = @async.with_timeout_opt(wait, () => {
    let count = self.flights[0].write_once(self.connections[0])
    if @async.is_being_cancelled() {
      @async.pause()
      raise luna_online_tcp_error(LunaOnlineTcpCancelled)
    }
    count
  }) catch {
    error if @async.is_being_cancelled() => {
      self.begin_disconnect(failure=LunaOnlineTcpCancelled)
      self.close_transport_on_reactor()
      raise error
    }
    _ => {
      self.begin_disconnect(failure=LunaOnlineTcpWrite)
      self.close_transport_on_reactor()
      return LunaOnlineTcpCleanupRequired
    }
  }
  let count = match written {
    None => {
      self.begin_disconnect(failure=LunaOnlineTcpTimeout)
      self.close_transport_on_reactor()
      return LunaOnlineTcpCleanupRequired
    }
    Some(value) => value
  }
  if count <= 0 || count > length {
    self.begin_disconnect(failure=LunaOnlineTcpWrite)
    self.close_transport_on_reactor()
    return LunaOnlineTcpCleanupRequired
  }
  let progress = self.offers[0].confirm(length=count) catch {
    _ => {
      self.begin_disconnect(failure=LunaOnlineTcpCoordinator)
      self.close_transport_on_reactor()
      return LunaOnlineTcpCleanupRequired
    }
  }
  // Confirmation invalidates the exact coordinator Offer before the backing
  // storage can return to the reusable mutable phase.
  try! self.flights[0].release()
  self.flights.clear()
  self.offers.clear()
  if progress == LunaOnlineFramedCoordinatorRejectionReady {
    return self.handle_rejection_on_reactor()
  }
  luna_online_tcp_map_coordinator_progress(self, progress)
}
