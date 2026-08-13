///|
/// Closes and reaps the current child before any recovery retirement.
pub fn WorkerService::begin_recovery(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  if self.lifecycle != ServiceReady && self.lifecycle != ServiceRecovering {
    raise Invalid(Lifecycle)
  }
  if self.process.lifecycle() == RootBoundRecovering {
    self.lifecycle = ServiceRecovering
    return
  }
  if self.process.lifecycle() != RootBoundReady &&
    self.process.lifecycle() != RootBoundRecoveryRequired &&
    self.process.lifecycle() != RootBoundRecoveryClosingFailed {
    raise Invalid(Lifecycle)
  }
  self.lifecycle = ServiceRecovering
  self.process.begin_recovery() catch {
    error => raise root_bound_process_error(error)
  }
}

///|
/// Retires the oldest closed-child obligation. A pinned valid completion is
/// committed normally; an unreturned plan is committed as the given failure.
pub fn WorkerService::recover_oldest(
  self : WorkerService,
  failure : @worker_protocol.WorkerFailure,
) -> UInt64 raise WorkerServiceError {
  self.require_lifecycle(ServiceRecovering)
  if self.process.lifecycle() != RootBoundRecovering {
    raise Invalid(Recovery)
  }
  let slot = self.oldest_slot()
  if slot < 0 {
    raise Invalid(Sequence)
  }
  let sequence = self.flight_sequences[slot]
  if self.flight_states[slot] == FLIGHT_RECEIVED {
    return self.commit_received_slot(slot)
  }
  let plan = self.scheduler.submitted_plan_for_sequence(sequence) catch {
    error => raise scheduler_error(error)
  }
  if self.flight_states[slot] == FLIGHT_SUBMITTED {
    let submission = self.process.submission_for_sequence(sequence) catch {
      error => raise root_bound_process_error(error)
    }
    self.process.validate_abandonment(submission) catch {
      error => raise root_bound_process_error(error)
    }
    self.scheduler.fail_plan(plan, failure) catch {
      error => raise scheduler_error(error)
    }
    self.process.abandon_submission(submission) catch {
      error => raise root_bound_process_error(error)
    }
  } else if self.flight_states[slot] == FLIGHT_PLANNED {
    self.scheduler.fail_plan(plan, failure) catch {
      error => raise scheduler_error(error)
    }
  } else {
    raise Invalid(Recovery)
  }
  self.clear_slot(slot)
  sequence
}

///|
pub fn WorkerService::restart(
  self : WorkerService,
) -> WorkerServiceRestart raise WorkerServiceError {
  self.require_lifecycle(ServiceRecovering)
  if self.outstanding_count() != 0 {
    raise Invalid(Recovery)
  }
  let startup = self.process.recovery_startup_contract() catch {
    error => raise root_bound_process_error(error)
  }
  validate_startup_contract(self.scheduler, self.binding, startup)
  let result = self.process.restart() catch {
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
  self.process.retry_replacement_cleanup() catch {
    error => raise root_bound_process_error(error)
  }
  self.lifecycle = ServiceRecovering
}

///|
pub fn WorkerService::shutdown_clean(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_lifecycle(ServiceReady)
  if self.outstanding_count() != 0 {
    raise Invalid(Lifecycle)
  }
  self.process.shutdown_clean() catch {
    error => {
      self.lifecycle = ServiceRecovering
      raise root_bound_process_error(error)
    }
  }
  self.process.close() catch {
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
  self.process.close() catch {
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
  if self.outstanding_count() != 0 {
    raise Invalid(Lifecycle)
  }
  self.process.close() catch {
    error => {
      self.lifecycle = ServiceClosingFailed
      raise root_bound_process_error(error)
    }
  }
  self.lifecycle = ServiceClosed
}
