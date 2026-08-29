///|
fn LunaOnlineFramedConnectionPool::free_slot(
  self : LunaOnlineFramedConnectionPool,
) -> Int {
  for offset in 0..<self.slots.length() {
    let index = (self.io_cursor + offset) % self.slots.length()
    if self.slots[index].phase == LUNA_FRAMED_POOL_FREE {
      return index
    }
  }
  -1
}

///|
fn LunaOnlineFramedConnectionSlot::activate(
  self : LunaOnlineFramedConnectionSlot,
  connection : @socket.Tcp,
) -> Bool {
  if self.phase != LUNA_FRAMED_POOL_FREE ||
    self.generation == 0xffffffffffffffffUL {
    return false
  }
  let receipt = self.receipt_workspace.begin() catch { _ => return false }
  let work = self.request_workspace.begin() catch {
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
  self.request_works.push(work)
  self.phase = LUNA_FRAMED_POOL_RECEIVING
  true
}

///|
async fn LunaOnlineFramedConnectionPool::accept_one(
  self : LunaOnlineFramedConnectionPool,
) -> LunaOnlineFramedConnectionPoolProgress {
  if self.listeners.length() != 1 ||
    self.active_connections >= self.slots.length() {
    return LunaOnlineFramedConnectionPoolBackpressured
  }
  let slot_index = self.free_slot()
  if slot_index < 0 {
    return LunaOnlineFramedConnectionPoolBackpressured
  }
  let accepted = @async.with_timeout_opt(
    self.pool_limits.io_poll_timeout_millis,
    () => self.listeners[0].accept(),
  ) catch {
    error if @async.is_being_cancelled() => raise error
    _ => {
      self.fail_and_drain()
      return LunaOnlineFramedConnectionPoolFailedProgress
    }
  }
  match accepted {
    None => LunaOnlineFramedConnectionPoolBackpressured
    Some((connection, _)) => {
      let mut installed = false
      defer (if !installed { connection.close() })
      if self.drain_requested || !self.slots[slot_index].activate(connection) {
        return LunaOnlineFramedConnectionPoolBackpressured
      }
      installed = true
      self.active_connections += 1
      self.network_accepts += 1UL
      self.telemetry.record_network_accept()
      self.io_cursor = (slot_index + 1) % self.slots.length()
      LunaOnlineFramedConnectionAccepted
    }
  }
}
