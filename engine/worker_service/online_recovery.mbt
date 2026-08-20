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
      self.request_phase = OnlineRequestTerminationPending
      service.lifecycle = ServiceRestartReady
      count
    }
    ExactDeviceAlreadyTerminal(generation) => {
      self.cut_source_generation = self.request_generation
      try! service.scheduler.advance_terminated_handle(handle, generation)
      self.request_generation = generation
      self.request_phase = OnlineRequestTerminationPending
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
      self.request_phase = OnlineRequestTerminationPending
      Remaining(count)
    }
    ExactLossComplete(generation) => {
      self.cut_source_generation = self.request_generation
      try! service.scheduler.advance_terminated_handle(handle, generation)
      self.request_generation = generation
      self.request_phase = OnlineRequestTerminationPending
      service.lifecycle = ServiceTerminalCloseReady
      Complete
    }
  }
}

///|
/// Advances one owner-resident terminal recovery transition after the worker
/// path failed during an online cut. No raw scheduler, process, handle, or
/// publication authority escapes. This cleanup operation may close/reap a
/// failed child and is therefore for the aggregate's off-reactor cleanup path.
pub fn OnlineWorkerLease::progress_terminal_recovery(
  self : OnlineWorkerLease,
  failure : @worker_protocol.WorkerFailure,
) -> OnlineTerminalRecoveryProgress raise WorkerServiceError {
  let service = self.require_current()
  match service.lifecycle {
    ServiceReady => {
      self.begin_recovery()
      OnlineTerminalRecoveryPending
    }
    ServiceRecovering => {
      if service.require_process().lifecycle() == RootBoundRecovering {
        ignore(self.recover_flight(failure))
      } else {
        self.begin_recovery()
      }
      OnlineTerminalRecoveryPending
    }
    ServiceDeviceInvalidationRequired => {
      ignore(self.invalidate_device_state())
      OnlineTerminalRecoveryPending
    }
    ServiceRestartReady => {
      self.abandon_restart()
      OnlineTerminalRecoveryPending
    }
    ServiceRestartForbidden => {
      ignore(self.drain_restart_forbidden())
      OnlineTerminalRecoveryPending
    }
    ServiceTerminalCloseReady => OnlineTerminalRecoveryPublicationReady
    _ => raise Invalid(Recovery)
  }
}

///|
/// Nonraising recovery disposition for an aggregate that already owns this
/// lease. A lower close/reap/invalidation failure retains authority and is
/// reported as retryable instead of allocating a post-mutation error.
pub fn OnlineWorkerLease::progress_terminal_recovery_status(
  self : OnlineWorkerLease,
  failure : @worker_protocol.WorkerFailure,
) -> OnlineTerminalRecoveryStatus {
  try self.progress_terminal_recovery(failure) catch {
    _ => OnlineTerminalRecoveryRetryable
  } noraise {
    OnlineTerminalRecoveryPending => OnlineTerminalRecoveryStillPending
    OnlineTerminalRecoveryPublicationReady => OnlineTerminalRecoveryReady
  }
}

///|
fn OnlineWorkerLease::preflight_terminal_request_retirement(
  self : OnlineWorkerLease,
  service : WorkerService,
  has_exchange_capacity : Bool,
) -> Unit raise WorkerServiceError {
  if self.request_phase != OnlineRequestTerminal ||
    self.cancel_reserved ||
    self.prepared_admission is Some(_) ||
    service.lifecycle != ServiceReady ||
    service.raw_outstanding_count() != 0 ||
    !service.scheduler.is_quiescent() ||
    !has_exchange_capacity {
    raise Invalid(Lifecycle)
  }
}

///|
fn OnlineWorkerLease::commit_terminal_request_retirement(
  self : OnlineWorkerLease,
) -> Unit {
  self.handle = None
  self.request_generation = @inference.RequestGeneration::initial()
  self.cut_source_generation = @inference.RequestGeneration::initial()
  self.expected_position = 0
  self.cancel_reserved = false
  self.request_phase = OnlineRequestEmpty
}

///|
/// Retires only the exact completed request while preserving the live worker,
/// scheduler history, lease epoch, ownership family, and publication cursor.
/// The complete preflight precedes the nonraising request-local reset.
pub fn OnlineWorkerLease::retire_terminal_request(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  let service = self.require_current()
  self.preflight_terminal_request_retirement(
    service,
    service.require_process().has_exchange_capacity(),
  )
  self.commit_terminal_request_retirement()
}

///|
/// Cleanly closes a healthy service only after its exact terminal publication
/// and every scheduler/process obligation have been retired.
pub fn OnlineWorkerLease::shutdown_clean(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  let service = self.require_current()
  if self.request_phase != OnlineRequestTerminal || self.cancel_reserved {
    raise Invalid(Lifecycle)
  }
  service.shutdown_clean_impl()
  self.request_phase = OnlineRequestRetired
  service.ownership_mode = OnlineRetiredOwnership
}

///|
/// Closes an online-family service before admission, including after a prior
/// session was released and a fresh epoch was claimed solely for cleanup.
pub fn OnlineWorkerLease::shutdown_clean_empty(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  let service = self.require_current()
  if self.request_phase != OnlineRequestEmpty ||
    self.cancel_reserved ||
    !service.scheduler.is_quiescent() {
    raise Invalid(Lifecycle)
  }
  service.shutdown_clean_impl()
  self.request_phase = OnlineRequestRetired
  service.ownership_mode = OnlineRetiredOwnership
}

///|
/// Force-closes only from an already terminally drained recovery phase.
pub fn OnlineWorkerLease::close_terminal(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  let service = self.require_current()
  if self.request_phase != OnlineRequestTerminal ||
    self.cancel_reserved ||
    service.lifecycle != ServiceTerminalCloseReady {
    raise Invalid(Lifecycle)
  }
  service.close_impl()
  self.request_phase = OnlineRequestRetired
  service.ownership_mode = OnlineRetiredOwnership
}

///|
/// Retries a terminal close without restoring raw authority.
pub fn OnlineWorkerLease::retry_close_terminal(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  let service = self.require_current()
  if self.request_phase != OnlineRequestTerminal || self.cancel_reserved {
    raise Invalid(Lifecycle)
  }
  match terminal_close_retry_action(service.lifecycle) {
    1 => service.close_impl()
    2 => service.retry_close_impl()
    _ => abort("terminal close retry action invariant")
  }
  self.request_phase = OnlineRequestRetired
  service.ownership_mode = OnlineRetiredOwnership
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
  if self.request_phase != OnlineRequestEmpty || self.cancel_reserved {
    raise Invalid(Lifecycle)
  }
  if service.lifecycle == ServiceReady {
    service.shutdown_clean_impl()
  } else if service.lifecycle == ServiceRecovering {
    service.close_impl()
  } else if service.lifecycle == ServiceClosingFailed {
    service.retry_close_impl()
  } else {
    raise Invalid(Lifecycle)
  }
  self.request_phase = OnlineRequestRetired
  service.ownership_mode = OnlineRetiredOwnership
}
