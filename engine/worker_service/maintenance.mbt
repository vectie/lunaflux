///|
fn WorkerService::require_no_maintenance(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  if self.maintenance_mode != MAINTENANCE_NONE {
    raise Invalid(Lifecycle)
  }
}

///|
fn WorkerService::begin_recovery_maintenance_impl(
  self : WorkerService,
  failure : @worker_protocol.WorkerFailure,
) -> Unit raise WorkerServiceError {
  self.require_no_maintenance()
  if self.lifecycle == ServiceRestartForbidden {
    self.maintenance_failure = failure
    self.maintenance_mode = MAINTENANCE_RECOVERY
    self.maintenance_phase = MAINTENANCE_DRAIN
    return
  }
  if self.lifecycle != ServiceReady && self.lifecycle != ServiceRecovering {
    raise Invalid(Lifecycle)
  }
  // A transport/process failure may synchronously close the child before the
  // aggregate selects cooperative maintenance. Resume from that exact retained
  // RootBoundRecovering owner instead of trying to begin child cleanup twice.
  if self.require_transport().is_recovering() {
    self.maintenance_failure = self
      .require_transport()
      .recovery_failure(failure)
    self.maintenance_mode = MAINTENANCE_RECOVERY
    if !self.require_transport().recovery_cleanup_complete() {
      self.lifecycle = ServiceRecovering
      self.maintenance_phase = MAINTENANCE_CHILD
    } else if self.flight is FlightEmpty && !self.has_pending_plan() {
      self.lifecycle = ServiceDeviceInvalidationRequired
      self.maintenance_phase = MAINTENANCE_INVALIDATE
    } else {
      self.lifecycle = ServiceRecovering
      self.maintenance_phase = MAINTENANCE_RETIRE
    }
    return
  }
  self.maintenance_failure = self
    .require_transport()
    .begin_recovery_maintenance(failure)
  self.maintenance_mode = MAINTENANCE_RECOVERY
  self.maintenance_phase = MAINTENANCE_CHILD
  self.lifecycle = ServiceRecovering
}

///|
fn WorkerService::begin_shutdown_maintenance_impl(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_no_maintenance()
  if self.lifecycle != ServiceReady &&
    self.lifecycle != ServiceTerminalCloseReady &&
    self.lifecycle != ServiceClosingFailed {
    raise Invalid(Lifecycle)
  }
  if self.raw_outstanding_count() != 0 ||
    self.scheduler.live_request_count() != 0 {
    raise Invalid(Lifecycle)
  }
  self.require_transport().begin_close_maintenance()
  self.maintenance_mode = MAINTENANCE_SHUTDOWN
  self.maintenance_phase = MAINTENANCE_CHILD
  self.lifecycle = ServiceClosingFailed
}

///|
fn WorkerService::begin_restart_cleanup_maintenance_impl(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_no_maintenance()
  self.require_lifecycle(ServiceReplacementCleanup)
  self.require_transport().begin_replacement_cleanup_maintenance()
  self.maintenance_mode = MAINTENANCE_REPLACEMENT_CLEANUP
  self.maintenance_phase = MAINTENANCE_CHILD
}

///|
fn WorkerService::progress_child_maintenance(
  self : WorkerService,
) -> LunaWorkerServiceMaintenanceProgress raise WorkerServiceError {
  let progress = self.require_transport().progress_maintenance()
  self.apply_child_maintenance_progress(
    progress.is_cleanup_required(),
    progress.is_cleanup_stuck(),
    progress.is_pending(),
    progress.is_advanced(),
    progress.is_complete(),
  )
}

///|
fn WorkerService::apply_child_maintenance_progress(
  self : WorkerService,
  cleanup_required : Bool,
  cleanup_stuck : Bool,
  pending : Bool,
  advanced : Bool,
  complete : Bool,
) -> LunaWorkerServiceMaintenanceProgress raise WorkerServiceError {
  if cleanup_required {
    return LunaWorkerServiceMaintenanceCleanupRequired
  }
  if cleanup_stuck {
    return LunaWorkerServiceMaintenanceCleanupStuck
  }
  if pending {
    return LunaWorkerServiceMaintenancePending
  }
  if advanced {
    return LunaWorkerServiceMaintenanceAdvanced
  }
  if !complete {
    raise Invalid(Recovery)
  }
  if self.maintenance_mode == MAINTENANCE_RECOVERY {
    self.maintenance_phase = MAINTENANCE_RETIRE
    LunaWorkerServiceRecoveryReady
  } else if self.maintenance_mode == MAINTENANCE_REPLACEMENT_CLEANUP {
    self.maintenance_mode = MAINTENANCE_NONE
    self.maintenance_phase = 0
    self.lifecycle = ServiceRestartReady
    LunaWorkerServiceRestartReady
  } else {
    self.maintenance_mode = MAINTENANCE_NONE
    self.maintenance_phase = 0
    self.lifecycle = ServiceClosed
    LunaWorkerServiceClosed
  }
}

///|
fn WorkerService::progress_recovery_retirement(
  self : WorkerService,
) -> LunaWorkerServiceMaintenanceProgress raise WorkerServiceError {
  if self.flight is FlightEmpty && !self.has_pending_plan() {
    self.lifecycle = ServiceDeviceInvalidationRequired
    self.maintenance_phase = MAINTENANCE_INVALIDATE
    return LunaWorkerServiceMaintenanceAdvanced
  }
  let failure = self.maintenance_failure
  try {
    match self.flight {
      FlightPendingFailure(_, _, _) =>
        ignore(self.recover_pending_failure(failure))
      FlightEmpty => {
        let sequence = self.pending_plan_sequence
        let plan = self.scheduler.submitted_plan_for_sequence(sequence) catch {
          error => raise scheduler_error(error)
        }
        self.scheduler.fail_plan(plan, failure) catch {
          error => raise scheduler_error(error)
        }
        self.retire_scheduler_only_failure(sequence)
        self.pending_plan_sequence = 0UL
        self.lifecycle = ServiceDeviceInvalidationRequired
      }
      _ => ignore(self.recover_physical_flight_impl(failure))
    }
  } catch {
    Scheduler(Invalid(issue=Backpressured, ..)) =>
      return LunaWorkerServiceMaintenancePending
    error => raise error
  }
  if self.lifecycle == ServiceRestartForbidden {
    self.maintenance_phase = MAINTENANCE_DRAIN
  } else if self.lifecycle == ServiceDeviceInvalidationRequired {
    self.maintenance_phase = MAINTENANCE_INVALIDATE
  }
  LunaWorkerServiceMaintenanceAdvanced
}

///|
fn WorkerService::progress_recovery_invalidation(
  self : WorkerService,
) -> LunaWorkerServiceMaintenanceProgress raise WorkerServiceError {
  ignore(self.invalidate_device_state_impl()) catch {
    Scheduler(Invalid(issue=Backpressured, ..)) =>
      return LunaWorkerServiceMaintenancePending
    error => raise error
  }
  self.maintenance_mode = MAINTENANCE_NONE
  self.maintenance_phase = 0
  LunaWorkerServiceRestartReady
}

///|
fn WorkerService::progress_recovery_drain(
  self : WorkerService,
) -> LunaWorkerServiceMaintenanceProgress raise WorkerServiceError {
  let result = self.drain_restart_forbidden_impl() catch {
    Scheduler(Invalid(issue=Backpressured, ..)) =>
      return LunaWorkerServiceMaintenancePending
    error => raise error
  }
  if result is Complete {
    self.maintenance_phase = MAINTENANCE_BEGIN_CLOSE
  }
  LunaWorkerServiceMaintenanceAdvanced
}

///|
fn WorkerService::progress_begin_terminal_close(
  self : WorkerService,
) -> LunaWorkerServiceMaintenanceProgress raise WorkerServiceError {
  self.require_transport().begin_close_maintenance()
  self.maintenance_mode = MAINTENANCE_SHUTDOWN
  self.maintenance_phase = MAINTENANCE_CHILD
  self.lifecycle = ServiceClosingFailed
  LunaWorkerServiceMaintenanceAdvanced
}

///|
fn WorkerService::progress_maintenance_impl(
  self : WorkerService,
) -> LunaWorkerServiceMaintenanceProgress raise WorkerServiceError {
  if self.maintenance_mode == MAINTENANCE_NONE {
    raise Invalid(Lifecycle)
  }
  match self.maintenance_phase {
    MAINTENANCE_CHILD => self.progress_child_maintenance()
    MAINTENANCE_RETIRE => self.progress_recovery_retirement()
    MAINTENANCE_INVALIDATE => self.progress_recovery_invalidation()
    MAINTENANCE_DRAIN => self.progress_recovery_drain()
    MAINTENANCE_BEGIN_CLOSE => self.progress_begin_terminal_close()
    _ => raise Invalid(Lifecycle)
  }
}

///|
fn WorkerService::maintenance_remaining_millis_impl(
  self : WorkerService,
) -> Int raise WorkerServiceError {
  if self.maintenance_mode == MAINTENANCE_NONE {
    raise Invalid(Lifecycle)
  }
  if self.maintenance_phase != MAINTENANCE_CHILD {
    return 0
  }
  self.require_transport().maintenance_remaining_millis()
}
