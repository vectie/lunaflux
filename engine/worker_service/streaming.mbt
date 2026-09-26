///|
fn WorkerService::progress_streaming(
  self : WorkerService,
) -> Bool raise WorkerServiceError {
  if !self.scheduler.streaming_enabled() {
    return false
  }
  guard self.require_transport() is RootedSingle(process) else {
    raise Invalid(WorkerEnvelope)
  }
  let now = process.streaming_now_millis() catch {
    error => raise physical_transport_error(error)
  }
  self.scheduler.observe_streaming_time(now) catch {
    error => raise scheduler_error(error)
  }
  if process.streaming_exchange_pending() {
    let reply = process.progress_streaming_exchange() catch {
      error => raise physical_transport_error(error)
    }
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
  process.begin_streaming_exchange(command) catch {
    error => raise physical_transport_error(error)
  }
  true
}

///|
fn WorkerService::bind_streaming_replacement(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  if !self.scheduler.streaming_enabled() {
    return
  }
  guard self.require_transport() is RootedSingle(process) else {
    raise Invalid(WorkerEnvelope)
  }
  self.scheduler.bind_streaming_replacement(
    process.startup_contract().streaming_epoch(),
  ) catch {
    error => raise scheduler_error(error)
  }
}
