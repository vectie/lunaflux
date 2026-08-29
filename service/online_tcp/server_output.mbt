///|
fn LunaOnlineTcpOutputWrite::copy_stream_event(
  self : LunaOnlineTcpOutputWrite,
  stream : @online_session.LunaOnlineFramedStream,
) -> @online_session.LunaOnlineFramedEventOffer raise {
  let scratch = self.require_mutating()
  stream.copy_framed_event_chunk_to(
    scratch.mutable,
    destination_offset=0,
    maximum_length=scratch.capacity,
  )
}

///|
async fn LunaOnlineTcpServer::write_event_on_reactor(
  self : LunaOnlineTcpServer,
) -> LunaOnlineTcpServerProgress {
  if self.connections.length() != 1 ||
    self.streams.length() != 1 ||
    self.offers.length() != 0 ||
    self.flights.length() != 0 ||
    !self.output.is_idle() {
    self.begin_connection_retirement(
      LunaLogInternal,
      failure=LunaOnlineTcpLifecycle,
    )
    return LunaOnlineTcpServerAdvanced
  }
  let write = self.output.begin_write() catch {
    _ => {
      self.begin_connection_retirement(
        LunaLogInternal,
        failure=LunaOnlineTcpLifecycle,
      )
      return LunaOnlineTcpServerAdvanced
    }
  }
  let offer = write.copy_stream_event(self.stream()) catch {
    _ => {
      try! write.abort()
      self.begin_connection_retirement(
        LunaLogInternal,
        failure=LunaOnlineTcpCoordinator,
      )
      return LunaOnlineTcpServerAdvanced
    }
  }
  let length = offer.length() catch {
    _ => {
      try! write.abort()
      self.begin_connection_retirement(
        LunaLogInternal,
        failure=LunaOnlineTcpCoordinator,
      )
      return LunaOnlineTcpServerAdvanced
    }
  }
  let flight = write.publish(length) catch {
    _ => {
      try! write.abort()
      self.begin_connection_retirement(
        LunaLogInternal,
        failure=LunaOnlineTcpLifecycle,
      )
      return LunaOnlineTcpServerAdvanced
    }
  }
  self.offers.push(offer)
  self.flights.push(flight)
  let wait = self.bounded_transport_wait(self.limits.write_timeout_millis) catch {
    _ => {
      self.begin_connection_retirement(
        LunaLogDeadline,
        failure=LunaOnlineTcpCoordinator,
      )
      return LunaOnlineTcpServerAdvanced
    }
  }
  if wait <= 0 {
    self.begin_connection_retirement(
      LunaLogDeadline,
      failure=LunaOnlineTcpTimeout,
    )
    return LunaOnlineTcpServerAdvanced
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
      self.begin_connection_retirement(
        LunaLogCancellation,
        failure=LunaOnlineTcpCancelled,
      )
      raise error
    }
    _ => {
      self.begin_connection_retirement(
        LunaLogNetworkWrite,
        failure=LunaOnlineTcpWrite,
      )
      return LunaOnlineTcpServerAdvanced
    }
  }
  let count = match written {
    None => {
      self.begin_connection_retirement(
        LunaLogDeadline,
        failure=LunaOnlineTcpTimeout,
      )
      return LunaOnlineTcpServerAdvanced
    }
    Some(value) => value
  }
  if count <= 0 || count > length {
    self.begin_connection_retirement(
      LunaLogNetworkWrite,
      failure=LunaOnlineTcpWrite,
    )
    return LunaOnlineTcpServerAdvanced
  }
  let progress = self.offers[0].confirm(length=count) catch {
    _ => {
      self.begin_connection_retirement(
        LunaLogInternal,
        failure=LunaOnlineTcpCoordinator,
      )
      return LunaOnlineTcpServerAdvanced
    }
  }
  // The exact Offer is consumed before the dual-view backing returns mutable.
  try! self.flights[0].release()
  self.flights.clear()
  self.offers.clear()
  self.map_stream_progress(progress)
}
