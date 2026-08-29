///|
async fn LunaOnlineOpenAIServer::accept_on_reactor(
  self : LunaOnlineOpenAIServer,
) -> LunaOnlineTcpServerProgress {
  if self.phase != LUNA_ONLINE_TCP_SERVER_LISTENING ||
    self.listeners.length() != 1 ||
    self.connections.length() != 0 ||
    self.streams.length() != 0 ||
    self.receipts.length() != 0 ||
    self.http_works.length() != 0 {
    self.latch_service_drain()
    return LunaOnlineTcpServerDrainingProgress
  }
  if self.connection_epoch == 0xffffffffffffffffUL {
    self.latch_service_drain()
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
    let receipt = self.receipt_workspace.begin() catch { _ => return false }
    let http = self.http_workspace.begin() catch {
      _ => {
        receipt.abort() catch {
          _ => ()
        }
        return false
      }
    }
    self.connections.push(connection)
    self.receipts.push(receipt)
    self.http_works.push(http)
    self.connection_epoch += 1UL
    self.connection_phase = LUNA_OPENAI_HTTP_RECEIVE
    self.phase = LUNA_ONLINE_TCP_SERVER_CONNECTED
    installed = true
    true
  }) catch {
    error if @async.is_being_cancelled() => {
      if self.connections.length() == 1 {
        self.connections[0].close()
        self.connections.clear()
      }
      self.clear_receipt()
      self.clear_http()
      self.latch_service_drain()
      raise error
    }
    _ => {
      self.latch_failure(LunaOnlineTcpAccept)
      self.latch_service_drain()
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
      self.latch_service_drain()
      LunaOnlineTcpServerDrainingProgress
    }
    None => {
      self.phase = LUNA_ONLINE_TCP_SERVER_LISTENING
      LunaOnlineTcpServerAcceptTimedOut
    }
  }
}
