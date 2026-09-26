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
priv struct StreamingSession {
  capability : @streaming_ir.Capability
  slots : FixedArray[HostSlot]
  mut control : @streaming_ir.Control
  mut flight : TransferFlight
}

///|
pub fn PagedGraphExecutor::prepare_streaming(
  self : PagedGraphExecutor,
  capability : @streaming_ir.Capability,
  worker : UInt64,
) -> Unit raise DeviceStepError {
  let descriptor = complete_descriptor(self.resources)
  let separate = descriptor.context.has_separate_host_memory() catch {
    error => raise ExecutorDevice(PersistentKvArena, error)
  }
  if !separate {
    raise InvalidExecutor(ResourceState)
  }
  if worker == 0UL ||
    self.resources.streaming is Some(_) ||
    capability.page_count() != descriptor.layout.total_page_count() {
    raise InvalidExecutor(ResourceState)
  }
  guard self.resources.activation_workspace is Some(workspace) && workspace.open else {
    raise InvalidExecutor(ResourceState)
  }
  if capability.workspace_budget() > workspace.resource.byte_count() {
    raise InvalidExecutor(ResourceState)
  }
  for i in 0..<capability.shape_count() {
    if capability.shape_at(i).tokens() > descriptor.limits.max_tokens {
      raise InvalidExecutor(ResourceState)
    }
  }
  let plan = @host_cache.Plan::new(
    descriptor.layout,
    slots=capability.slots(),
    transfer_lanes=1,
    host_budget=capability.host_bytes(),
    separate_host_memory=true,
  ) catch {
    _ => raise InvalidExecutor(ResourceState)
  }
  if plan.slot_bytes() != capability.page_bytes() {
    raise InvalidExecutor(ResourceState)
  }
  self.prepare_host_cache(plan)
  self.resources.streaming = Some({
    capability,
    slots: FixedArray::make(capability.slots(), Empty),
    control: @streaming_ir.Control::new(worker),
    flight: Idle,
  })
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
pub fn PagedGraphExecutor::streaming_command(
  self : PagedGraphExecutor,
  command : @streaming_ir.Command,
) -> @streaming_ir.Reply raise DeviceStepError {
  require_paged_lifecycle(self.lifecycle, GraphReady)
  guard self.resources.streaming is Some(session) else {
    raise InvalidExecutor(ResourceState)
  }
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
        ignore(self.cancel_prefix_transfer(operation))
      }
      let progress = self.poll_prefix_transfer(operation)
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
        if !self.evict_host_prefix(entry) {
          return rejected
        }
        session.slots[command.host.slot] = Empty
        return { command, status: Evicted, host: command.host, }
      }
      let start = match command.opcode {
        Spill => self.begin_prefix_spill(command.page)
        Restore => {
          guard session.entry(command.host) is Stored(entry) else {
            return rejected
          }
          self.begin_prefix_restore(entry, command.page)
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
