///|
async fn LunaOnlineTcpServer::accept_on_reactor(
  self : LunaOnlineTcpServer,
) -> LunaOnlineTcpServerProgress {
  if self.phase == LUNA_ONLINE_TCP_SERVER_ACCEPTING ||
    self.phase != LUNA_ONLINE_TCP_SERVER_LISTENING ||
    self.listeners.length() != 1 ||
    self.connections.length() != 0 ||
    self.streams.length() != 0 {
    self.latch_failure(LunaOnlineTcpLifecycle)
    self.drain_requested = true
    self.close_listener_pending = self.listeners.length() == 1
    return LunaOnlineTcpServerDrainingProgress
  }
  self.phase = LUNA_ONLINE_TCP_SERVER_ACCEPTING
  let accepted = @async.with_timeout_opt(self.limits.accept_timeout_millis, () => {
    let (connection, _) = self.listeners[0].accept()
    let mut installed = false
    defer (if !installed { connection.close() })
    if @async.is_being_cancelled() {
      @async.pause()
      raise luna_online_tcp_error(LunaOnlineTcpCancelled)
    }
    if self.phase != LUNA_ONLINE_TCP_SERVER_ACCEPTING ||
      self.connections.length() != 0 ||
      self.streams.length() != 0 {
      return false
    }
    // The native socket is committed before the exact Stream is opened. There
    // is no await between these owner mutations.
    self.connections.push(connection)
    installed = true
    let opened = self.service().open_stream()
    if opened.kind() != LunaOnlineFramedOpened {
      connection.close()
      self.connections.clear()
      return false
    }
    let stream = self.service().take_open_stream(opened) catch {
        _ => {
          connection.close()
          self.connections.clear()
          self.service().begin_drain()
          self.service_drain_started = true
          return false
        }
      }
    self.streams.push(stream)
    self.phase = LUNA_ONLINE_TCP_SERVER_CONNECTED
    true
  }) catch {
    error if @async.is_being_cancelled() => {
      if self.connections.length() == 1 {
        self.connections[0].close()
        self.connections.clear()
      }
      if self.listeners.length() == 1 {
        self.listeners[0].close()
        self.listeners.clear()
      }
      if self.streams.length() == 1 {
        self.streams[0].disconnect() catch {
          _ => ()
        }
      }
      self.service().begin_drain()
      self.service_drain_started = true
      self.drain_requested = true
      self.phase = LUNA_ONLINE_TCP_SERVER_DRAINING
      raise error
    }
    _ => {
      self.latch_failure(LunaOnlineTcpAccept)
      self.drain_requested = true
      self.close_listener_pending = self.listeners.length() == 1
      self.phase = LUNA_ONLINE_TCP_SERVER_DRAINING
      return LunaOnlineTcpServerDrainingProgress
    }
  }
  match accepted {
    Some(true) => {
      self.failure_rule = 0
      self.connection_telemetry_active = true
      self.record_network_accept()
      LunaOnlineTcpServerConnectedProgress
    }
    Some(false) => {
      self.latch_failure(LunaOnlineTcpLifecycle)
      self.drain_requested = true
      self.close_listener_pending = self.listeners.length() == 1
      self.phase = LUNA_ONLINE_TCP_SERVER_DRAINING
      LunaOnlineTcpServerDrainingProgress
    }
    None => {
      self.phase = LUNA_ONLINE_TCP_SERVER_LISTENING
      LunaOnlineTcpServerAcceptTimedOut
    }
  }
}

///|
async fn LunaOnlineTcpServer::read_on_reactor(
  self : LunaOnlineTcpServer,
) -> LunaOnlineTcpServerProgress {
  if self.connections.length() != 1 ||
    self.streams.length() != 1 ||
    self.tail_length != 0 {
    self.begin_connection_retirement(
      LunaLogInternal,
      failure=LunaOnlineTcpLifecycle,
    )
    return LunaOnlineTcpServerAdvanced
  }
  let wait = self.bounded_transport_wait(self.limits.input_idle_timeout_millis) catch {
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
      self.begin_connection_retirement(
        LunaLogCancellation,
        failure=LunaOnlineTcpCancelled,
      )
      raise error
    }
    _ => {
      self.begin_connection_retirement(
        LunaLogNetworkRead,
        failure=LunaOnlineTcpRead,
      )
      return LunaOnlineTcpServerAdvanced
    }
  }
  match received {
    None => {
      self.begin_connection_retirement(
        LunaLogDeadline,
        failure=LunaOnlineTcpTimeout,
      )
      LunaOnlineTcpServerAdvanced
    }
    Some(count) => {
      if count == 0 {
        self.begin_connection_retirement(LunaLogPeerClosed)
        return LunaOnlineTcpServerAdvanced
      }
      if count < 0 || count > self.input.length() {
        self.begin_connection_retirement(
          LunaLogNetworkRead,
          failure=LunaOnlineTcpRead,
        )
        return LunaOnlineTcpServerAdvanced
      }
      self.tail_start = 0
      self.tail_length = count
      self.tail_backpressured = false
      self.backpressure_episode_active = false
      self.request_receipt_complete = false
      LunaOnlineTcpServerAdvanced
    }
  }
}

///|
fn LunaOnlineTcpServer::offer_tail(
  self : LunaOnlineTcpServer,
) -> LunaOnlineTcpServerProgress {
  if self.tail_length <= 0 ||
    self.tail_start < 0 ||
    self.tail_start > self.input.length() - self.tail_length {
    self.begin_connection_retirement(
      LunaLogInternal,
      failure=LunaOnlineTcpLifecycle,
    )
    return LunaOnlineTcpServerAdvanced
  }
  let offered = self
    .stream()
    .offer_luna_framed(
      self.input,
      source_offset=self.tail_start,
      length=self.tail_length,
    ) catch {
      _ => {
        self.begin_connection_retirement(
          LunaLogValidation,
          failure=LunaOnlineTcpCoordinator,
        )
        return LunaOnlineTcpServerAdvanced
      }
    }
  let consumed = offered.consumed_bytes()
  if consumed < 0 || consumed > self.tail_length {
    self.begin_connection_retirement(
      LunaLogInternal,
      failure=LunaOnlineTcpCoordinator,
    )
    return LunaOnlineTcpServerAdvanced
  }
  match offered.kind() {
    LunaOnlineFramedIngressAccepted => {
      if consumed <= 0 {
        self.begin_connection_retirement(
          LunaLogInternal,
          failure=LunaOnlineTcpCoordinator,
        )
        return LunaOnlineTcpServerAdvanced
      }
      self.tail_start += consumed
      self.tail_length -= consumed
      self.tail_backpressured = false
      self.backpressure_episode_active = false
      if self.tail_length == 0 {
        self.tail_start = 0
      }
      let complete = self.stream().luna_framed_receipt_complete() catch {
          _ => {
            self.begin_connection_retirement(
              LunaLogInternal,
              failure=LunaOnlineTcpCoordinator,
            )
            return LunaOnlineTcpServerAdvanced
          }
        }
      // Max-one stops reading after the first complete receipt. Pipeline mode
      // keeps the exact Stream and socket live so a retained/coalesced tail or
      // the next read can begin another bounded frame. Its boundary evidence
      // starts only when the authenticated Admission observation is recorded.
      self.request_receipt_complete = complete && !self.pipeline_mode
      LunaOnlineTcpServerAdvanced
    }
    LunaOnlineFramedIngressBackpressured => {
      if consumed != 0 {
        self.begin_connection_retirement(
          LunaLogInternal,
          failure=LunaOnlineTcpCoordinator,
        )
        return LunaOnlineTcpServerAdvanced
      }
      self.tail_backpressured = true
      if !self.backpressure_episode_active {
        self.backpressure_episode_active = true
        self.record_backpressure()
      }
      LunaOnlineTcpServerBackpressured
    }
    LunaOnlineFramedIngressDraining => {
      self.begin_connection_retirement(LunaLogServiceDraining)
      LunaOnlineTcpServerAdvanced
    }
  }
}
