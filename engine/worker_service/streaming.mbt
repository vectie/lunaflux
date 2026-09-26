///|
fn WorkerService::progress_streaming(
  self : WorkerService,
) -> Bool raise WorkerServiceError {
  if !self.scheduler.streaming_enabled() {
    return false
  }
  let transport = self.require_transport()
  errdefer (if transport is TensorParallelGroup(group) {
    group.fail_streaming()
  })
  let now = transport.streaming_now_millis()
  self.scheduler.observe_streaming_time(now) catch {
    error => raise scheduler_error(error)
  }
  if transport.streaming_exchange_pending() {
    let reply = transport.progress_streaming_exchange()
    if reply is Some(reply) {
      self.scheduler.accept_streaming_reply(reply) catch {
        error => raise scheduler_error(error)
      }
    }
    return true
  }
  let command = self.scheduler.next_streaming_command(now) catch {
    error => raise scheduler_error(error)
  }
  guard command is Some(command) else { return false }
  transport.begin_streaming_exchange(command)
  true
}

///|
fn WorkerService::bind_streaming_replacement(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  if !self.scheduler.streaming_enabled() {
    return
  }
  let epoch = self.require_transport().streaming_epoch()
  self.scheduler.bind_streaming_replacement(epoch) catch {
    error => raise scheduler_error(error)
  }
}

///|
fn WorkerPhysicalTransport::streaming_epoch(
  self : WorkerPhysicalTransport,
) -> UInt64 raise WorkerServiceError {
  match self {
    RootedSingle(process) => process.startup_contract().streaming_epoch()
    TensorParallelGroup(group) =>
      group.streaming_epoch() catch {
        error => raise tensor_parallel_transport_error(error)
      }
    _ => raise Invalid(WorkerEnvelope)
  }
}

///|
fn WorkerPhysicalTransport::streaming_now_millis(
  self : WorkerPhysicalTransport,
) -> UInt64 raise WorkerServiceError {
  match self {
    RootedSingle(process) =>
      process.streaming_now_millis() catch {
        error => raise physical_transport_error(error)
      }
    TensorParallelGroup(group) =>
      group.streaming_now_millis() catch {
        error => raise tensor_parallel_transport_error(error)
      }
    _ => raise Invalid(WorkerEnvelope)
  }
}

///|
fn WorkerPhysicalTransport::streaming_exchange_pending(
  self : WorkerPhysicalTransport,
) -> Bool {
  match self {
    RootedSingle(process) => process.streaming_exchange_pending()
    TensorParallelGroup(group) => group.streaming_exchange_pending()
    _ => false
  }
}

///|
fn WorkerPhysicalTransport::begin_streaming_exchange(
  self : WorkerPhysicalTransport,
  command : @streaming_ir.Command,
) -> Unit raise WorkerServiceError {
  match self {
    RootedSingle(process) =>
      process.begin_streaming_exchange(command) catch {
        error => raise physical_transport_error(error)
      }
    TensorParallelGroup(group) =>
      group.begin_streaming_exchange(command) catch {
        error => raise tensor_parallel_transport_error(error)
      }
    _ => raise Invalid(WorkerEnvelope)
  }
}

///|
fn WorkerPhysicalTransport::progress_streaming_exchange(
  self : WorkerPhysicalTransport,
) -> @streaming_ir.ReplyOption raise WorkerServiceError {
  match self {
    RootedSingle(process) =>
      process.progress_streaming_exchange() catch {
        error => raise physical_transport_error(error)
      }
    TensorParallelGroup(group) =>
      group.progress_streaming_exchange() catch {
        error => raise tensor_parallel_transport_error(error)
      }
    _ => raise Invalid(WorkerEnvelope)
  }
}
