///|
async fn LunaOnlineTcpEndpoint::accept_on_reactor(
  self : LunaOnlineTcpEndpoint,
) -> LunaOnlineTcpProgress {
  if self.phase == LUNA_ONLINE_TCP_ACCEPTING {
    self.begin_disconnect(failure=LunaOnlineTcpCancelled)
    self.close_transport_on_reactor()
    return LunaOnlineTcpCleanupRequired
  }
  if self.phase != LUNA_ONLINE_TCP_LISTENING || self.servers.length() != 1 {
    self.begin_disconnect(failure=LunaOnlineTcpLifecycle)
    self.close_transport_on_reactor()
    return LunaOnlineTcpCleanupRequired
  }
  self.phase = LUNA_ONLINE_TCP_ACCEPTING
  let accepted = @async.with_timeout_opt(self.limits.accept_timeout_millis, () => {
    let (connection, _) = self.servers[0].accept()
    let mut installed = false
    defer (if !installed { connection.close() })
    if @async.is_being_cancelled() {
      @async.pause()
      raise luna_online_tcp_error(LunaOnlineTcpCancelled)
    }
    if self.phase != LUNA_ONLINE_TCP_ACCEPTING || !self.connections.is_empty() {
      return false
    }
    self.connections.push(connection)
    self.servers[0].close()
    self.servers.clear()
    self.phase = LUNA_ONLINE_TCP_CONNECTED
    installed = true
    true
  }) catch {
    error if @async.is_being_cancelled() => {
      self.begin_disconnect(failure=LunaOnlineTcpCancelled)
      self.close_transport_on_reactor()
      raise error
    }
    _ => {
      self.begin_disconnect(failure=LunaOnlineTcpAccept)
      self.close_transport_on_reactor()
      return LunaOnlineTcpCleanupRequired
    }
  }
  match accepted {
    Some(true) => LunaOnlineTcpConnectedProgress
    Some(false) => {
      self.begin_disconnect(failure=LunaOnlineTcpAccept)
      self.close_transport_on_reactor()
      LunaOnlineTcpCleanupRequired
    }
    None => {
      self.begin_disconnect(failure=LunaOnlineTcpTimeout)
      self.close_transport_on_reactor()
      LunaOnlineTcpCleanupRequired
    }
  }
}

///|
async fn LunaOnlineTcpEndpoint::read_on_reactor(
  self : LunaOnlineTcpEndpoint,
) -> LunaOnlineTcpProgress {
  if self.connections.length() != 1 || self.tail_length != 0 {
    self.begin_disconnect(failure=LunaOnlineTcpLifecycle)
    self.close_transport_on_reactor()
    return LunaOnlineTcpCleanupRequired
  }
  let wait = self.bounded_transport_wait(self.limits.input_idle_timeout_millis) catch {
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
  let received = @async.with_timeout_opt(wait, () => {
    let count = self.connections[0].read(
      self.input,
      offset=0,
      max_len=self.input.length(),
    )
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
      self.begin_disconnect(failure=LunaOnlineTcpRead)
      self.close_transport_on_reactor()
      return LunaOnlineTcpCleanupRequired
    }
  }
  match received {
    None => {
      self.begin_disconnect(failure=LunaOnlineTcpTimeout)
      self.close_transport_on_reactor()
      LunaOnlineTcpCleanupRequired
    }
    Some(count) => {
      if count == 0 {
        self.begin_disconnect()
        self.close_transport_on_reactor()
        return LunaOnlineTcpCleanupRequired
      }
      if count > self.input.length() {
        self.begin_disconnect(failure=LunaOnlineTcpRead)
        self.close_transport_on_reactor()
        return LunaOnlineTcpCleanupRequired
      }
      self.tail_start = 0
      self.tail_length = count
      self.tail_backpressured = false
      LunaOnlineTcpAdvanced
    }
  }
}

///|
fn LunaOnlineTcpEndpoint::offer_tail(
  self : LunaOnlineTcpEndpoint,
) -> LunaOnlineTcpProgress {
  if self.tail_length <= 0 ||
    self.tail_start < 0 ||
    self.tail_start > self.input.length() - self.tail_length {
    self.begin_disconnect(failure=LunaOnlineTcpLifecycle)
    return LunaOnlineTcpCleanupRequired
  }
  let offered = self
    .coordinator()
    .offer_luna_framed(
      self.input,
      source_offset=self.tail_start,
      length=self.tail_length,
    ) catch {
      _ => {
        self.begin_disconnect(failure=LunaOnlineTcpCoordinator)
        return LunaOnlineTcpCleanupRequired
      }
    }
  let consumed = offered.consumed_bytes()
  if consumed < 0 || consumed > self.tail_length {
    self.begin_disconnect(failure=LunaOnlineTcpCoordinator)
    return LunaOnlineTcpCleanupRequired
  }
  match offered.kind() {
    LunaOnlineFramedIngressAccepted => {
      if consumed <= 0 {
        self.begin_disconnect(failure=LunaOnlineTcpCoordinator)
        return LunaOnlineTcpCleanupRequired
      }
      self.tail_start += consumed
      self.tail_length -= consumed
      self.tail_backpressured = false
      if self.tail_length == 0 {
        self.tail_start = 0
      }
      LunaOnlineTcpAdvanced
    }
    LunaOnlineFramedIngressBackpressured => {
      if consumed != 0 {
        self.begin_disconnect(failure=LunaOnlineTcpCoordinator)
        return LunaOnlineTcpCleanupRequired
      }
      self.tail_backpressured = true
      LunaOnlineTcpBackpressured
    }
    LunaOnlineFramedIngressDraining => {
      if consumed != 0 {
        self.begin_disconnect(failure=LunaOnlineTcpCoordinator)
        return LunaOnlineTcpCleanupRequired
      }
      self.tail_start = 0
      self.tail_length = 0
      self.tail_backpressured = false
      self.drain_requested = true
      self.phase = LUNA_ONLINE_TCP_DRAINING
      LunaOnlineTcpCleanupRequired
    }
  }
}
