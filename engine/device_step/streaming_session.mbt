///|
#valtype
priv enum HostSlot {
  Empty
  Stored(@residency.Entry)
}

///|
#valtype
priv enum TransferFlight {
  Idle
  Flying(@streaming_ir.Command, @host_cache.Operation)
}

///|
pub struct StreamingSession {
  priv cache : @host_cache.Cache
  priv capability : @streaming_ir.Capability
  priv slots : FixedArray[HostSlot]
  priv mut control : @streaming_ir.Control
  priv mut flight : TransferFlight
}

///|
fn StreamingSession::entry(
  self : StreamingSession,
  id : @streaming_ir.HostPage,
) -> HostSlot {
  if id.slot < 0 || id.slot >= self.slots.length() {
    return Empty
  }
  match self.slots[id.slot] {
    Stored(entry) if entry.generation() == id.generation => Stored(entry)
    _ => Empty
  }
}

///|
/// Interpret one bounded scalar command. Polling never waits for CUDA.
pub fn StreamingSession::command(
  self : StreamingSession,
  command : @streaming_ir.Command,
) -> @streaming_ir.Reply raise DeviceStepError {
  let session = self
  let cache = self.cache
  let rejected : @streaming_ir.Reply = {
    command,
    status: Rejected,
    host: command.host,
  }
  let decision = session.control.decide(command, session.capability)
  if decision.action is Reject {
    return rejected
  }
  session.control = decision.next
  match decision.action {
    PollTransfer | CancelTransfer => {
      guard session.flight is Flying(original, operation) else {
        return rejected
      }
      if !command.same_transaction(original) {
        return rejected
      }
      if command.opcode is Cancel {
        ignore(cache.cancel(operation))
      }
      let progress = cache.poll(operation) catch {
        error => raise ExecutorDevice(PersistentKvArena, error)
      }
      let host : @streaming_ir.HostPage = {
        slot: operation.entry().slot(),
        generation: operation.entry().generation(),
      }
      match progress {
        Pending => { command, status: Pending, host, }
        HostReady => {
          session.slots[host.slot] = Stored(operation.entry())
          session.flight = Idle
          session.control = session.control.completed()
          { command, status: HostReady, host, }
        }
        DeviceReady => {
          session.flight = Idle
          session.control = session.control.completed()
          { command, status: DeviceReady, host, }
        }
        Cancelled => {
          session.flight = Idle
          session.control = session.control.completed()
          { command, status: Cancelled, host, }
        }
        Stale => raise InvalidExecutor(ResourceState)
      }
    }
    _ => {
      if command.opcode is Evict {
        guard session.entry(command.host) is Stored(entry) else {
          return rejected
        }
        if !cache.evict(entry) {
          return rejected
        }
        session.slots[command.host.slot] = Empty
        return { command, status: Evicted, host: command.host, }
      }
      let start = match command.opcode {
        Spill =>
          cache.begin_spill(command.page) catch {
            error => raise ExecutorDevice(PersistentKvArena, error)
          }
        Restore => {
          guard session.entry(command.host) is Stored(entry) else {
            return rejected
          }
          cache.begin_restore(entry, command.page) catch {
            error => raise ExecutorDevice(PersistentKvArena, error)
          }
        }
        _ => return rejected
      }
      match start {
        Started(operation) => {
          session.flight = Flying(command, operation)
          { command, status: Pending, host: command.host, }
        }
        _ => rejected
      }
    }
  }
}

///|
pub fn StreamingSession::new(
  capability : @streaming_ir.Capability,
  worker : UInt64,
  cache : @host_cache.Cache,
) -> StreamingSession {
  {
    cache,
    capability,
    slots: FixedArray::make(capability.slots(), Empty),
    control: @streaming_ir.Control::new(worker),
    flight: Idle,
  }
}
