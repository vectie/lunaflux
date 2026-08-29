///|
/// Closes and reaps the current child before any recovery retirement.
fn WorkerService::begin_recovery_impl(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  if self.lifecycle != ServiceReady && self.lifecycle != ServiceRecovering {
    raise Invalid(Lifecycle)
  }
  if self.require_transport().is_recovering() {
    self.lifecycle = if self.flight is FlightEmpty && !self.has_pending_plan() {
      ServiceDeviceInvalidationRequired
    } else {
      ServiceRecovering
    }
    return
  }
  if !self.require_transport().allows_begin_recovery() {
    raise Invalid(Lifecycle)
  }
  self.lifecycle = ServiceRecovering
  self.require_transport().begin_recovery()
  if self.flight is FlightEmpty && !self.has_pending_plan() {
    self.lifecycle = ServiceDeviceInvalidationRequired
  }
}

///|
fn WorkerService::recover_pending_failure(
  self : WorkerService,
  failure : @worker_protocol.WorkerFailure,
) -> UInt64 raise WorkerServiceError {
  guard self.flight
    is FlightPendingFailure(
      pending_sequence,
      original_sequence,
      restart_forbidden
    ) else {
    raise Invalid(Recovery)
  }
  let plan = self.scheduler.submitted_plan_for_sequence(pending_sequence) catch {
    error => raise scheduler_error(error)
  }
  self.scheduler.fail_plan(plan, failure) catch {
    error => raise scheduler_error(error)
  }
  self.retire_scheduler_only_failure(pending_sequence)
  self.clear_flight()
  self.lifecycle = if restart_forbidden {
    ServiceRestartForbidden
  } else {
    ServiceDeviceInvalidationRequired
  }
  original_sequence
}

///|
fn WorkerService::finish_recovered_flight(
  self : WorkerService,
  sequence : UInt64,
  restart_forbidden : Bool,
) -> UInt64 {
  self.clear_flight()
  if self.has_pending_plan() {
    let pending_sequence = self.pending_plan_sequence
    self.pending_plan_sequence = 0UL
    self.flight = FlightPendingFailure(
      pending_sequence, sequence, restart_forbidden,
    )
    return sequence
  }
  self.lifecycle = if restart_forbidden {
    ServiceRestartForbidden
  } else {
    ServiceDeviceInvalidationRequired
  }
  sequence
}

///|
/// Retires only the physical side of the exact closed-child obligation. A
/// pending scheduler-only B side remains a distinct later obligation.
fn WorkerService::recover_physical_flight_impl(
  self : WorkerService,
  failure : @worker_protocol.WorkerFailure,
) -> UInt64 raise WorkerServiceError {
  self.require_lifecycle(ServiceRecovering)
  if !self.require_transport().is_recovering() {
    raise Invalid(Recovery)
  }
  if !self.require_transport().recovery_cleanup_complete() {
    raise PhysicalTransport(PhysicalBackpressured)
  }
  let sequence = self.flight_sequence_value()
  if self.require_transport().has_received(sequence) {
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
        self.scheduler.replace_submitted_completion_with_failure(plan, failure) catch {
          error => raise scheduler_error(error)
        }
        self.require_transport().retire_received_failure(sequence)
        return self.finish_recovered_flight(sequence, false)
      }
    } noraise {
      progress =>
        match progress {
          ServiceBackpressured =>
            raise Scheduler(Invalid(field=OutputCapacity, issue=Backpressured))
          ServiceCommitted(committed) =>
            return self.finish_recovered_flight(committed, false)
          _ => abort("recovery completion progress invariant")
        }
    }
  }
  let plan = self.scheduler.submitted_plan_for_sequence(sequence) catch {
    error => raise scheduler_error(error)
  }
  if !self.require_transport().has_submission(sequence) {
    self.scheduler.fail_plan(plan, failure) catch {
      error => raise scheduler_error(error)
    }
    return self.finish_recovered_flight(sequence, true)
  }
  self.scheduler.fail_plan(plan, failure) catch {
    error => raise scheduler_error(error)
  }
  self.require_transport().abandon_submission(sequence)
  self.finish_recovered_flight(sequence, false)
}

///|
/// Blocking compatibility driver. Cooperative maintenance calls the physical
/// and pending obligation helpers on separate host turns.
fn WorkerService::recover_flight_impl(
  self : WorkerService,
  fallback_failure : @worker_protocol.WorkerFailure,
) -> UInt64 raise WorkerServiceError {
  self.require_lifecycle(ServiceRecovering)
  if !self.require_transport().is_recovering() {
    raise Invalid(Recovery)
  }
  let failure = self.require_transport().recovery_failure(fallback_failure)
  match self.flight {
    FlightPendingFailure(_, _, _) =>
      return self.recover_pending_failure(failure)
    FlightEmpty if self.has_pending_plan() => {
      let sequence = self.pending_plan_sequence
      self.pending_plan_sequence = 0UL
      self.flight = FlightPendingFailure(sequence, sequence, false)
      return self.recover_pending_failure(failure)
    }
    _ => ()
  }
  let sequence = self.recover_physical_flight_impl(failure)
  if self.flight is FlightPendingFailure(_, _, _) {
    ignore(self.recover_pending_failure(failure))
  }
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
  self
  .require_transport()
  .validate_recovery_binding(
    self.scheduler,
    self.identity,
    self.physical_binding,
  )
  let result = self.require_transport().restart()
  match result {
    PhysicalRestarted => {
      self.graph_telemetry.begin_replacement_child()
      self.lifecycle = ServiceReady
      Restarted
    }
    PhysicalRestartCleanupRequired => {
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
  self.require_transport().retry_replacement_cleanup()
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
  let shutdown_result : Result[Unit, WorkerServiceError] = Ok(
    self.require_transport().shutdown_clean(),
  ) catch {
    error => Err(error)
  }
  match shutdown_result {
    Err(error) => {
      self.lifecycle = ServiceRecovering
      raise error
    }
    Ok(_) => ()
  }
  let close_result : Result[Unit, WorkerServiceError] = Ok(
    self.require_transport().close(),
  ) catch {
    error => Err(error)
  }
  match close_result {
    Err(error) => {
      self.lifecycle = ServiceClosingFailed
      raise error
    }
    Ok(_) => ()
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
  self.require_transport().close()
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
  let close_result : Result[Unit, WorkerServiceError] = Ok(
    self.require_transport().close(),
  ) catch {
    error => Err(error)
  }
  match close_result {
    Err(error) => {
      self.lifecycle = ServiceClosingFailed
      raise error
    }
    Ok(_) => ()
  }
  self.lifecycle = ServiceClosed
}
