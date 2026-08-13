///|
/// Closes the failed child while retaining the exact online request and lease
/// epoch. Only this authenticated lease may enter online recovery.
pub fn OnlineWorkerLease::begin_recovery(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  ignore(self.require_pending_handle())
  self.require_current().begin_recovery_impl()
}

///|
/// Retires the exact failed flight without exposing its scheduler submission.
pub fn OnlineWorkerLease::recover_flight(
  self : OnlineWorkerLease,
  failure : @worker_protocol.WorkerFailure,
) -> UInt64 raise WorkerServiceError {
  self.require_current().recover_flight_impl(failure)
}

///|
/// Invalidates all device-resident state after the failed child is closed.
pub fn OnlineWorkerLease::invalidate_device_state(
  self : OnlineWorkerLease,
) -> Int raise WorkerServiceError {
  let service = self.require_current()
  let handle = self.require_pending_handle()
  service.require_lifecycle(ServiceDeviceInvalidationRequired)
  let result = service.scheduler.invalidate_device_state_exact(handle) catch {
    error => raise scheduler_error(error)
  }
  match result {
    ExactDeviceNoChange => {
      service.lifecycle = ServiceRestartReady
      0
    }
    ExactDeviceInvalidated(count, generation) => {
      self.cut_source_generation = self.request_generation
      try! service.scheduler.advance_terminated_handle(handle, generation)
      self.request_generation = generation
      self.request_phase = ONLINE_REQUEST_TERMINATION_PENDING
      service.lifecycle = ServiceRestartReady
      count
    }
    ExactDeviceAlreadyTerminal(generation) => {
      self.cut_source_generation = self.request_generation
      try! service.scheduler.advance_terminated_handle(handle, generation)
      self.request_generation = generation
      self.request_phase = ONLINE_REQUEST_TERMINATION_PENDING
      service.lifecycle = ServiceRestartReady
      0
    }
  }
}

///|
/// Starts the retained exact replacement only after device invalidation.
pub fn OnlineWorkerLease::restart(
  self : OnlineWorkerLease,
) -> WorkerServiceRestart raise WorkerServiceError {
  self.require_current().restart_impl()
}

///|
/// Retries cleanup of the exact failed replacement child.
pub fn OnlineWorkerLease::retry_restart_cleanup(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  self.require_current().retry_restart_cleanup_impl()
}

///|
/// Permanently selects deterministic terminal drain instead of replacement.
pub fn OnlineWorkerLease::abandon_restart(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  self.require_current().abandon_restart_impl()
}

///|
/// Drains one request after replacement became unsafe. A terminal notice for
/// this one-active-request online owner remains available through
/// `take_publication` before terminal close.
pub fn OnlineWorkerLease::drain_restart_forbidden(
  self : OnlineWorkerLease,
) -> @core.InstanceLossDrain raise WorkerServiceError {
  let service = self.require_current()
  let handle = self.require_pending_handle()
  service.require_lifecycle(ServiceRestartForbidden)
  let exact = service.scheduler.drain_instance_loss_exact(handle) catch {
    error => raise scheduler_error(error)
  }
  match exact {
    ExactLossRemaining(count, generation) => {
      self.cut_source_generation = self.request_generation
      try! service.scheduler.advance_terminated_handle(handle, generation)
      self.request_generation = generation
      self.request_phase = ONLINE_REQUEST_TERMINATION_PENDING
      Remaining(count)
    }
    ExactLossComplete(generation) => {
      self.cut_source_generation = self.request_generation
      try! service.scheduler.advance_terminated_handle(handle, generation)
      self.request_generation = generation
      self.request_phase = ONLINE_REQUEST_TERMINATION_PENDING
      service.lifecycle = ServiceTerminalCloseReady
      Complete
    }
  }
}

///|
/// Cleanly closes a healthy service only after its exact terminal publication
/// and every scheduler/process obligation have been retired.
pub fn OnlineWorkerLease::shutdown_clean(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  let service = self.require_current()
  if self.request_phase != ONLINE_REQUEST_TERMINAL || self.cancel_reserved {
    raise Invalid(Lifecycle)
  }
  service.shutdown_clean_impl()
  self.request_phase = ONLINE_REQUEST_RETIRED
  service.ownership_mode = SERVICE_MODE_ONLINE_RETIRED
}

///|
/// Closes an online-family service before admission, including after a prior
/// session was released and a fresh epoch was claimed solely for cleanup.
pub fn OnlineWorkerLease::shutdown_clean_empty(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  let service = self.require_current()
  if self.request_phase != ONLINE_REQUEST_EMPTY ||
    self.cancel_reserved ||
    !service.scheduler.is_quiescent() {
    raise Invalid(Lifecycle)
  }
  service.shutdown_clean_impl()
  self.request_phase = ONLINE_REQUEST_RETIRED
  service.ownership_mode = SERVICE_MODE_ONLINE_RETIRED
}

///|
/// Force-closes only from an already terminally drained recovery phase.
pub fn OnlineWorkerLease::close_terminal(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  let service = self.require_current()
  if self.request_phase != ONLINE_REQUEST_TERMINAL ||
    self.cancel_reserved ||
    service.lifecycle != ServiceTerminalCloseReady {
    raise Invalid(Lifecycle)
  }
  service.close_impl()
  self.request_phase = ONLINE_REQUEST_RETIRED
  service.ownership_mode = SERVICE_MODE_ONLINE_RETIRED
}

///|
/// Retries a terminal close without restoring raw authority.
pub fn OnlineWorkerLease::retry_close_terminal(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  let service = self.require_current()
  if self.request_phase != ONLINE_REQUEST_TERMINAL || self.cancel_reserved {
    raise Invalid(Lifecycle)
  }
  match terminal_close_retry_action(service.lifecycle) {
    1 => service.close_impl()
    2 => service.retry_close_impl()
    _ => abort("terminal close retry action invariant")
  }
  self.request_phase = ONLINE_REQUEST_RETIRED
  service.ownership_mode = SERVICE_MODE_ONLINE_RETIRED
}

///|
fn terminal_close_retry_action(
  lifecycle : WorkerServiceLifecycle,
) -> Int raise WorkerServiceError {
  if lifecycle == ServiceRecovering {
    1
  } else if lifecycle == ServiceClosingFailed {
    2
  } else {
    raise Invalid(Lifecycle)
  }
}

///|
/// Retries a failed empty-session clean close without granting raw authority.
pub fn OnlineWorkerLease::retry_close_empty(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  let service = self.require_current()
  if self.request_phase != ONLINE_REQUEST_EMPTY || self.cancel_reserved {
    raise Invalid(Lifecycle)
  }
  if service.lifecycle == ServiceRecovering {
    service.close_impl()
  } else if service.lifecycle == ServiceClosingFailed {
    service.retry_close_impl()
  } else {
    raise Invalid(Lifecycle)
  }
  self.request_phase = ONLINE_REQUEST_RETIRED
  service.ownership_mode = SERVICE_MODE_ONLINE_RETIRED
}
