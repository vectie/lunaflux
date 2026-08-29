///|
/// Closes the failed child while retaining the exact online request and lease
/// epoch. Only this authenticated lease may enter online recovery.
pub fn OnlineWorkerLease::begin_recovery(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  let service = self.require_current()
  if self.active_lane_count == 0 {
    raise Invalid(Lifecycle)
  }
  service.require_no_maintenance()
  service.begin_recovery_impl()
}

///|
/// Retires the exact failed flight without exposing its scheduler submission.
pub fn OnlineWorkerLease::recover_flight(
  self : OnlineWorkerLease,
  failure : @worker_protocol.WorkerFailure,
) -> UInt64 raise WorkerServiceError {
  let service = self.require_current()
  service.require_no_maintenance()
  service.recover_flight_impl(failure)
}

///|
/// Invalidates all device-resident state after the failed child is closed.
pub fn OnlineWorkerLease::invalidate_device_state(
  self : OnlineWorkerLease,
) -> Int raise WorkerServiceError {
  let service = self.require_current()
  service.require_no_maintenance()
  self.invalidate_device_state_exact_impl(service)
}

///|
fn OnlineWorkerLease::invalidate_device_state_exact_impl(
  self : OnlineWorkerLease,
  service : WorkerService,
) -> Int raise WorkerServiceError {
  service.require_lifecycle(ServiceDeviceInvalidationRequired)
  if self.active_lane_count == 0 {
    raise Invalid(Recovery)
  }
  for lane in 0..<self.lane_phases.length() {
    if self.lane_phases[lane] != ONLINE_LANE_EMPTY &&
      self.lane_restart_dispositions[lane] != 0 {
      raise Invalid(Recovery)
    }
  }
  let invalidated = service.scheduler.invalidate_device_state() catch {
    error => raise scheduler_error(error)
  }
  for lane in 0..<self.lane_phases.length() {
    if self.lane_phases[lane] == ONLINE_LANE_LIVE {
      self.lane_cut_source_generations[lane] = self.lane_request_generations[lane]
      self.lane_phases[lane] = ONLINE_LANE_TERMINATION_PENDING
      self.lane_restart_dispositions[lane] = 1
    } else if self.lane_phases[lane] == ONLINE_LANE_TERMINATION_PENDING {
      self.lane_restart_dispositions[lane] = 2
    }
  }
  service.lifecycle = ServiceRestartReady
  invalidated
}

///|
/// Converts the exact request retained across a successful worker restart into
/// one authenticated WorkerFailed terminal. Active requests already own that
/// terminal after device invalidation; a waiting request is terminalized only
/// after the replacement worker is proven ready. Semantic retention remains
/// live until ordinary terminal-request retirement.
pub fn OnlineWorkerLease::prepare_restarted_failure_terminal(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  if !self.progress_restarted_failure_terminals() {
    raise Invalid(Recovery)
  }
}

///|
/// Converts at most one waiting request retained across a successful restart
/// into an authenticated WorkerFailed terminal. False retains all lane and
/// scheduler authority for the next cooperative call.
pub fn OnlineWorkerLease::progress_restarted_failure_terminals(
  self : OnlineWorkerLease,
) -> Bool raise WorkerServiceError {
  let service = self.require_current()
  service.require_no_maintenance()
  service.require_lifecycle(ServiceReady)
  if service.raw_outstanding_count() != 0 {
    raise Invalid(Recovery)
  }
  for lane in 0..<self.lane_phases.length() {
    if self.lane_phases[lane] != ONLINE_LANE_EMPTY &&
      self.lane_restart_dispositions[lane] == 0 {
      raise Invalid(Recovery)
    }
  }
  if service.scheduler.live_request_count() == 0 {
    return true
  }
  let progress = service.scheduler.drain_instance_loss() catch {
    error => raise scheduler_error(error)
  }
  match progress {
    Remaining(_) => false
    Complete => true
  }
}

///|
/// Retries cleanup of the exact failed replacement child.
pub fn OnlineWorkerLease::retry_restart_cleanup(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  let service = self.require_current()
  service.require_no_maintenance()
  service.retry_restart_cleanup_impl()
}

///|
/// Starts bounded cleanup of a failed replacement while retaining this exact
/// online lease and request generation.
pub fn OnlineWorkerLease::begin_restart_cleanup_maintenance(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  self.require_current().begin_restart_cleanup_maintenance_impl()
}

///|
/// Permanently selects deterministic terminal drain instead of replacement.
pub fn OnlineWorkerLease::abandon_restart(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  let service = self.require_current()
  service.require_no_maintenance()
  service.abandon_restart_impl()
}

///|
/// Drains one request after replacement became unsafe. A terminal notice for
/// this one-active-request online owner remains available through
/// `take_publication` before terminal close.
pub fn OnlineWorkerLease::drain_restart_forbidden(
  self : OnlineWorkerLease,
) -> @core.InstanceLossDrain raise WorkerServiceError {
  let service = self.require_current()
  service.require_no_maintenance()
  self.drain_restart_forbidden_exact_impl(service)
}

///|
fn OnlineWorkerLease::drain_restart_forbidden_exact_impl(
  self : OnlineWorkerLease,
  service : WorkerService,
) -> @core.InstanceLossDrain raise WorkerServiceError {
  let request = self.compatibility_request()
  let lane = request.lane
  let handle = request.require_pending_handle()
  service.require_lifecycle(ServiceRestartForbidden)
  // Device invalidation may already have terminalized the sole active
  // request. Its queued WorkerFailed publication owns the fresh generation;
  // `take_publication` authenticates that terminal and advances the retained
  // lane. Do not try to drain or advance the old handle twice.
  if service.scheduler.live_request_count() == 0 {
    service.lifecycle = ServiceTerminalCloseReady
    return Complete
  }
  let exact = service.scheduler.drain_instance_loss_exact(handle) catch {
    error => raise scheduler_error(error)
  }
  match exact {
    ExactLossRemaining(count, generation) => {
      self.lane_cut_source_generations[lane] = self.lane_request_generations[lane]
      try! service.scheduler.advance_terminated_handle(handle, generation)
      self.lane_request_generations[lane] = generation
      self.lane_phases[lane] = ONLINE_LANE_TERMINATION_PENDING
      Remaining(count)
    }
    ExactLossComplete(generation) => {
      self.lane_cut_source_generations[lane] = self.lane_request_generations[lane]
      try! service.scheduler.advance_terminated_handle(handle, generation)
      self.lane_request_generations[lane] = generation
      self.lane_phases[lane] = ONLINE_LANE_TERMINATION_PENDING
      service.lifecycle = ServiceTerminalCloseReady
      Complete
    }
  }
}

///|
fn OnlineWorkerLease::drain_restart_forbidden_progress(
  self : OnlineWorkerLease,
  service : WorkerService,
) -> @core.InstanceLossDrain raise WorkerServiceError {
  service.require_lifecycle(ServiceRestartForbidden)
  let progress = service.scheduler.drain_instance_loss() catch {
    error => raise scheduler_error(error)
  }
  for lane in 0..<self.lane_phases.length() {
    if self.lane_phases[lane] == ONLINE_LANE_LIVE {
      self.lane_cut_source_generations[lane] = self.lane_request_generations[lane]
      self.lane_phases[lane] = ONLINE_LANE_TERMINATION_PENDING
    }
  }
  if progress is Complete {
    service.lifecycle = ServiceTerminalCloseReady
  }
  progress
}

///|
/// Starts cooperative recovery for the exact authenticated online request.
pub fn OnlineWorkerLease::begin_recovery_maintenance(
  self : OnlineWorkerLease,
  failure : @worker_protocol.WorkerFailure,
) -> Unit raise WorkerServiceError {
  if self.active_lane_count == 0 {
    raise Invalid(Lifecycle)
  }
  self.require_current().begin_recovery_maintenance_impl(failure)
}

///|
/// Starts cooperative terminal close without releasing semantic retention
/// before child and root retirement complete.
pub fn OnlineWorkerLease::begin_shutdown_maintenance(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  let service = self.require_current()
  if self.cancel_lane >= 0 || self.active_lane_count > 1 {
    raise Invalid(Lifecycle)
  }
  if self.active_lane_count == 1 {
    let request = self.compatibility_request()
    if self.lane_phases[request.lane] != ONLINE_LANE_TERMINAL ||
      !self.lane_semantic_retentions[request.lane].is_live() {
      raise Invalid(Lifecycle)
    }
  } else if self.active_lane_count == 0 {
    for lane in 0..<self.lane_semantic_retentions.length() {
      if self.lane_semantic_retentions[lane].is_live() {
        raise Invalid(Lifecycle)
      }
    }
  }
  service.begin_shutdown_maintenance_impl()
}

///|
/// Advances one authenticated online maintenance transition. Exact request
/// generation changes remain in this lease and never escape as raw handles.
pub fn OnlineWorkerLease::progress_maintenance(
  self : OnlineWorkerLease,
) -> LunaWorkerServiceMaintenanceProgress raise WorkerServiceError {
  let service = self.require_current()
  if service.maintenance_mode == MAINTENANCE_RECOVERY &&
    service.maintenance_phase == MAINTENANCE_INVALIDATE {
    ignore(self.invalidate_device_state_exact_impl(service)) catch {
      Scheduler(Invalid(issue=Backpressured, ..)) =>
        return LunaWorkerServiceMaintenancePending
      error => raise error
    }
    service.maintenance_mode = MAINTENANCE_NONE
    service.maintenance_phase = 0
    return LunaWorkerServiceRestartReady
  }
  if service.maintenance_mode == MAINTENANCE_RECOVERY &&
    service.maintenance_phase == MAINTENANCE_DRAIN {
    let result = self.drain_restart_forbidden_progress(service) catch {
      Scheduler(Invalid(issue=Backpressured, ..)) =>
        return LunaWorkerServiceMaintenancePending
      error => raise error
    }
    if result is Complete {
      service.maintenance_mode = MAINTENANCE_NONE
      service.maintenance_phase = 0
      return LunaWorkerServiceFailCloseReady
    }
    return LunaWorkerServiceMaintenanceAdvanced
  }
  let progress = service.progress_maintenance_impl()
  if progress is LunaWorkerServiceClosed {
    if self.active_lane_count == 1 {
      let lane = self.compatibility_request().lane
      if self.lane_phases[lane] == ONLINE_LANE_TERMINAL {
        self.release_lane(lane)
      }
    }
    service.ownership_mode = OnlineRetiredOwnership
  }
  progress
}

///|
pub fn OnlineWorkerLease::maintenance_remaining_millis(
  self : OnlineWorkerLease,
) -> Int raise WorkerServiceError {
  self.require_current().maintenance_remaining_millis_impl()
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
  if service.maintenance_mode != MAINTENANCE_NONE {
    let progress = self.progress_maintenance()
    match progress {
      LunaWorkerServiceFailCloseReady =>
        return OnlineTerminalRecoveryPublicationReady
      LunaWorkerServiceRestartReady => {
        self.abandon_restart()
        return OnlineTerminalRecoveryPending
      }
      _ => return OnlineTerminalRecoveryPending
    }
  }
  match service.lifecycle {
    ServiceReady => {
      self.begin_recovery_maintenance(failure)
      OnlineTerminalRecoveryPending
    }
    ServiceRecovering => {
      self.begin_recovery_maintenance(failure)
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
