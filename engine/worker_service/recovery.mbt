///|
/// Closes and reaps the current child before any recovery retirement.
fn WorkerService::begin_recovery_impl(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  if self.lifecycle != ServiceReady && self.lifecycle != ServiceRecovering {
    raise Invalid(Lifecycle)
  }
  if self.require_process().lifecycle() == RootBoundRecovering {
    self.lifecycle = match self.flight {
      FlightEmpty => ServiceDeviceInvalidationRequired
      _ => ServiceRecovering
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
  match self.flight {
    FlightEmpty => self.lifecycle = ServiceDeviceInvalidationRequired
    _ => ()
  }
}

///|
/// Retires the exact closed-child obligation. A pinned valid completion is
/// committed normally; an unreturned plan is committed as the given failure.
/// If no process reservation exists, scheduler state is still retired but this
/// owner becomes permanently restart-forbidden.
fn WorkerService::recover_flight_impl(
  self : WorkerService,
  failure : @worker_protocol.WorkerFailure,
) -> UInt64 raise WorkerServiceError {
  self.require_lifecycle(ServiceRecovering)
  if self.require_process().lifecycle() != RootBoundRecovering {
    raise Invalid(Recovery)
  }
  let sequence = self.flight_sequence_value()
  try self.require_process().received_for_sequence(sequence) catch {
    RootBoundFailure(Process(Invalid(Completion))) => ()
    error => raise root_bound_process_error(error)
  } noraise {
    _ => {
      match self.flight {
        FlightAccepted(_) => ()
        _ => self.flight = FlightReceived(sequence)
      }
      try self.commit_received() catch {
        Scheduler(Invalid(issue=Backpressured, ..)) as error => raise error
        _ => {
          let plan = self.scheduler.submitted_plan_for_sequence(sequence) catch {
            error => raise scheduler_error(error)
          }
          let received = self.require_process().received_for_sequence(sequence) catch {
              error => raise root_bound_process_error(error)
            }
          self.require_process().validate_retirement(received) catch {
            error => raise root_bound_process_error(error)
          }
          self.scheduler.replace_submitted_completion_with_failure(
            plan, failure,
          ) catch {
            error => raise scheduler_error(error)
          }
          try! self.require_process().retire_received(received)
          self.clear_flight()
          self.lifecycle = ServiceDeviceInvalidationRequired
          return sequence
        }
      } noraise {
        progress =>
          match progress {
            ServiceBackpressured =>
              raise Scheduler(
                Invalid(field=OutputCapacity, issue=Backpressured),
              )
            ServiceCommitted(committed) => {
              self.lifecycle = ServiceDeviceInvalidationRequired
              return committed
            }
            _ => abort("recovery completion progress invariant")
          }
      }
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
      try! self.require_process().abandon_submission(submission)
    }
  }
  self.clear_flight()
  self.lifecycle = ServiceDeviceInvalidationRequired
  sequence
}

///|
fn WorkerService::invalidate_device_state_impl(
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
fn WorkerService::drain_restart_forbidden_impl(
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
fn WorkerService::abandon_restart_impl(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_lifecycle(ServiceRestartReady)
  self.lifecycle = ServiceRestartForbidden
}

///|
fn WorkerService::restart_impl(
  self : WorkerService,
) -> WorkerServiceRestart raise WorkerServiceError {
  self.require_lifecycle(ServiceRestartReady)
  if self.raw_outstanding_count() != 0 {
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
fn WorkerService::retry_restart_cleanup_impl(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_lifecycle(ServiceReplacementCleanup)
  self.require_process().retry_replacement_cleanup() catch {
    error => raise root_bound_process_error(error)
  }
  self.lifecycle = ServiceRestartReady
}

///|
fn WorkerService::shutdown_clean_impl(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_lifecycle(ServiceReady)
  if self.raw_outstanding_count() != 0 ||
    self.scheduler.live_request_count() != 0 {
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
fn WorkerService::retry_close_impl(
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
fn WorkerService::close_impl(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  if self.lifecycle == ServiceClosed {
    return
  }
  if self.raw_outstanding_count() != 0 ||
    self.scheduler.live_request_count() != 0 {
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
