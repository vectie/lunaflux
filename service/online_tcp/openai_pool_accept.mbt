///|
fn LunaOnlineOpenAIConnectionPool::free_slot(
  self : LunaOnlineOpenAIConnectionPool,
) -> Int {
  for offset in 0..<self.slots.length() {
    let index = (self.io_cursor + offset) % self.slots.length()
    if self.slots[index].phase == LUNA_OPENAI_POOL_FREE {
      return index
    }
  }
  -1
}

///|
fn LunaOnlineOpenAIConnectionSlot::activate(
  self : LunaOnlineOpenAIConnectionSlot,
  connection : @socket.Tcp,
) -> Bool {
  if self.phase != LUNA_OPENAI_POOL_FREE ||
    self.generation == 0xffffffffffffffffUL {
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
  self.generation += 1UL
  self.connections.push(connection)
  self.receipts.push(receipt)
  self.http_works.push(http)
  self.phase = LUNA_OPENAI_POOL_HTTP_RECEIVE
  true
}

///|
async fn LunaOnlineOpenAIConnectionPool::accept_one(
  self : LunaOnlineOpenAIConnectionPool,
) -> LunaOnlineOpenAIConnectionPoolProgress {
  if self.listeners.length() != 1 ||
    self.active_connections >= self.slots.length() {
    return LunaOnlineOpenAIConnectionPoolBackpressured
  }
  let slot_index = self.free_slot()
  if slot_index < 0 {
    return LunaOnlineOpenAIConnectionPoolBackpressured
  }
  let accepted = @async.with_timeout_opt(
    self.pool_limits.io_poll_timeout_millis,
    () => self.listeners[0].accept(),
  ) catch {
    error if @async.is_being_cancelled() => raise error
    _ => {
      self.begin_drain()
      return LunaOnlineOpenAIConnectionPoolDrainingProgress
    }
  }
  match accepted {
    None => LunaOnlineOpenAIConnectionPoolBackpressured
    Some((connection, _)) => {
      let mut installed = false
      defer (if !installed { connection.close() })
      if self.drain_requested || !self.slots[slot_index].activate(connection) {
        return LunaOnlineOpenAIConnectionPoolBackpressured
      }
      installed = true
      self.active_connections += 1
      self.telemetry.record_network_accept()
      self.io_cursor = (slot_index + 1) % self.slots.length()
      LunaOnlineOpenAIConnectionAccepted
    }
  }
}
