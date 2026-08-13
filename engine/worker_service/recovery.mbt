///|
/// Closes and reaps the current child before any recovery retirement.
pub fn WorkerService::begin_recovery(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  if self.lifecycle != ServiceReady && self.lifecycle != ServiceRecovering {
    raise Invalid(Lifecycle)
  }
  if self.require_process().lifecycle() == RootBoundRecovering {
    self.lifecycle = if self.flight_state == FLIGHT_EMPTY {
      ServiceDeviceInvalidationRequired
    } else {
      ServiceRecovering
    }
    return
  }
  if self.require_process().lifecycle() != RootBoundReady &&
    self.require_process().lifecycle() != RootBoundRecoveryRequired &&
    self.require_process().lifecycle() != RootBoundRecoveryClosingFailed {
    raise Invalid(Lifecycle)
  }
  self.lifecycle = ServiceRecovering
  self.require_process().begin_recovery() catch {
    error => raise root_bound_process_error(error)
  }
  if self.flight_state == FLIGHT_EMPTY {
    self.lifecycle = ServiceDeviceInvalidationRequired
  }
}

///|
/// Retires the exact closed-child obligation. A pinned valid completion is
/// committed normally; an unreturned plan is committed as the given failure.
/// If no process reservation exists, scheduler state is still retired but this
/// owner becomes permanently restart-forbidden.
pub fn WorkerService::recover_flight(
  self : WorkerService,
  failure : @worker_protocol.WorkerFailure,
) -> UInt64 raise WorkerServiceError {
  self.require_lifecycle(ServiceRecovering)
  if self.require_process().lifecycle() != RootBoundRecovering {
    raise Invalid(Recovery)
  }
  if self.flight_state == FLIGHT_EMPTY {
    raise Invalid(Sequence)
  }
  let sequence = self.flight_sequence
  try self.require_process().received_for_sequence(sequence) catch {
    RootBoundFailure(Process(Invalid(Completion))) => ()
    error => raise root_bound_process_error(error)
  } noraise {
    _ => {
      self.flight_state = FLIGHT_RECEIVED
      let committed = self.commit_received()
      self.lifecycle = ServiceDeviceInvalidationRequired
      return committed
    }
  }
  let plan = self.scheduler.submitted_plan_for_sequence(sequence) catch {
    error => raise scheduler_error(error)
  }
  try self.require_process().submission_for_sequence(sequence) catch {
    RootBoundFailure(Process(Invalid(Submission))) => {
      self.scheduler.fail_plan(plan, failure) catch {
        error => raise scheduler_error(error)
      }
      self.clear_flight()
      self.lifecycle = ServiceRestartForbidden
      return sequence
    }
    error => raise root_bound_process_error(error)
  } noraise {
    submission => {
      self.require_process().validate_abandonment(submission) catch {
        error => raise root_bound_process_error(error)
      }
      self.scheduler.fail_plan(plan, failure) catch {
        error => raise scheduler_error(error)
      }
      self.require_process().abandon_submission(submission) catch {
        error => raise root_bound_process_error(error)
      }
    }
  }
  self.clear_flight()
  self.lifecycle = ServiceDeviceInvalidationRequired
  sequence
}

///|
pub fn WorkerService::invalidate_device_state(
  self : WorkerService,
) -> Int raise WorkerServiceError {
  self.require_lifecycle(ServiceDeviceInvalidationRequired)
  let count = self.scheduler.invalidate_device_state() catch {
    error => raise scheduler_error(error)
  }
  self.lifecycle = ServiceRestartReady
  count
}

///|
/// Deterministically terminally retires one live request after reservation
/// ambiguity made replacement unsafe. `Complete` enables terminal close.
pub fn WorkerService::drain_restart_forbidden(
  self : WorkerService,
) -> @core.InstanceLossDrain raise WorkerServiceError {
  self.require_lifecycle(ServiceRestartForbidden)
  let progress = self.scheduler.drain_instance_loss() catch {
    error => raise scheduler_error(error)
  }
  if progress is Complete {
    self.lifecycle = ServiceTerminalCloseReady
  }
  progress
}

///|
/// Permanently abandons replacement from a restart-ready state. Retained
/// waiting work must then be terminally drained before close.
pub fn WorkerService::abandon_restart(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_lifecycle(ServiceRestartReady)
  self.lifecycle = ServiceRestartForbidden
}

///|
pub fn WorkerService::restart(
  self : WorkerService,
) -> WorkerServiceRestart raise WorkerServiceError {
  self.require_lifecycle(ServiceRestartReady)
  if self.outstanding_count() != 0 {
    raise Invalid(Recovery)
  }
  let startup = self.require_process().recovery_startup_contract() catch {
      error => raise root_bound_process_error(error)
    }
  validate_startup_contract(self.scheduler, self.binding, startup)
  let result = self.require_process().restart() catch {
      error => raise root_bound_process_error(error)
    }
  match result {
    RootBoundRestarted => {
      self.lifecycle = ServiceReady
      Restarted
    }
    RootBoundRestartCleanupRequired => {
      self.lifecycle = ServiceReplacementCleanup
      RestartCleanupRequired
    }
  }
}

///|
/// Retries cleanup of the exact failed replacement child retained by the
/// root-bound process owner. A later restart remains impossible until this
/// succeeds.
pub fn WorkerService::retry_restart_cleanup(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_lifecycle(ServiceReplacementCleanup)
  self.require_process().retry_replacement_cleanup() catch {
    error => raise root_bound_process_error(error)
  }
  self.lifecycle = ServiceRestartReady
}

///|
pub fn WorkerService::shutdown_clean(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_lifecycle(ServiceReady)
  if self.outstanding_count() != 0 || self.scheduler.live_request_count() != 0 {
    raise Invalid(Lifecycle)
  }
  self.require_process().shutdown_clean() catch {
    error => {
      self.lifecycle = ServiceRecovering
      raise root_bound_process_error(error)
    }
  }
  self.require_process().close() catch {
    error => {
      self.lifecycle = ServiceClosingFailed
      raise root_bound_process_error(error)
    }
  }
  self.lifecycle = ServiceClosed
}

///|
/// Retries terminal child/root retirement after `shutdown_clean` could not
/// consume every authority. This never starts recovery or replacement.
pub fn WorkerService::retry_close(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_lifecycle(ServiceClosingFailed)
  self.require_process().close() catch {
    error => raise root_bound_process_error(error)
  }
  self.lifecycle = ServiceClosed
}

///|
/// Force-retires a service with no outstanding scheduler obligations from a
/// recovery, failed-replacement-cleanup, or terminal-cleanup phase. The
/// root-bound owner still attempts child and root cleanup independently.
pub fn WorkerService::close(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  if self.lifecycle == ServiceClosed {
    return
  }
  if self.outstanding_count() != 0 || self.scheduler.live_request_count() != 0 {
    raise Invalid(Lifecycle)
  }
  if self.lifecycle == ServiceRestartForbidden {
    raise Invalid(Lifecycle)
  }
  self.require_process().close() catch {
    error => {
      self.lifecycle = ServiceClosingFailed
      raise root_bound_process_error(error)
    }
  }
  self.lifecycle = ServiceClosed
}
