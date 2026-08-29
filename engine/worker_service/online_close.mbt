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
  let request = self.compatibility_request()
  if self.lane_phases[request.lane] != ONLINE_LANE_TERMINAL ||
    self.cancel_lane >= 0 ||
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
  self.release_lane(self.compatibility_lane)
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
    service.require_transport().has_exchange_capacity(),
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
  service.require_no_maintenance()
  let request = self.compatibility_request()
  if self.lane_phases[request.lane] != ONLINE_LANE_TERMINAL ||
    self.cancel_lane >= 0 {
    raise Invalid(Lifecycle)
  }
  service.shutdown_clean_impl()
  self.release_lane(request.lane)
  service.ownership_mode = OnlineRetiredOwnership
}

///|
/// Closes an online-family service before admission, including after a prior
/// session was released and a fresh epoch was claimed solely for cleanup.
pub fn OnlineWorkerLease::shutdown_clean_empty(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  let service = self.require_current()
  service.require_no_maintenance()
  if self.active_lane_count != 0 ||
    self.cancel_lane >= 0 ||
    !service.scheduler.is_quiescent() {
    raise Invalid(Lifecycle)
  }
  service.shutdown_clean_impl()
  service.ownership_mode = OnlineRetiredOwnership
}

///|
/// Force-closes only from an already terminally drained recovery phase.
pub fn OnlineWorkerLease::close_terminal(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  let service = self.require_current()
  service.require_no_maintenance()
  let request = self.compatibility_request()
  if self.lane_phases[request.lane] != ONLINE_LANE_TERMINAL ||
    self.cancel_lane >= 0 ||
    service.lifecycle != ServiceTerminalCloseReady {
    raise Invalid(Lifecycle)
  }
  service.close_impl()
  self.release_lane(request.lane)
  service.ownership_mode = OnlineRetiredOwnership
}

///|
/// Retries a terminal close without restoring raw authority.
pub fn OnlineWorkerLease::retry_close_terminal(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  let service = self.require_current()
  service.require_no_maintenance()
  let request = self.compatibility_request()
  if self.lane_phases[request.lane] != ONLINE_LANE_TERMINAL ||
    self.cancel_lane >= 0 {
    raise Invalid(Lifecycle)
  }
  match terminal_close_retry_action(service.lifecycle) {
    1 => service.close_impl()
    2 => service.retry_close_impl()
    _ => abort("terminal close retry action invariant")
  }
  self.release_lane(request.lane)
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
  service.require_no_maintenance()
  if self.active_lane_count != 0 || self.cancel_lane >= 0 {
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
  service.ownership_mode = OnlineRetiredOwnership
}
