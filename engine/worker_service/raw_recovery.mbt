///|
pub fn WorkerService::begin_recovery(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_raw_mode()
  self.require_no_maintenance()
  self.begin_recovery_impl()
}

///|
pub fn WorkerService::recover_flight(
  self : WorkerService,
  failure : @worker_protocol.WorkerFailure,
) -> UInt64 raise WorkerServiceError {
  self.require_raw_mode()
  self.require_no_maintenance()
  self.recover_flight_impl(failure)
}

///|
pub fn WorkerService::invalidate_device_state(
  self : WorkerService,
) -> Int raise WorkerServiceError {
  self.require_raw_mode()
  self.require_no_maintenance()
  self.invalidate_device_state_impl()
}

///|
pub fn WorkerService::drain_restart_forbidden(
  self : WorkerService,
) -> @core.InstanceLossDrain raise WorkerServiceError {
  self.require_raw_mode()
  self.require_no_maintenance()
  self.drain_restart_forbidden_impl()
}

///|
pub fn WorkerService::abandon_restart(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_raw_mode()
  self.require_no_maintenance()
  self.abandon_restart_impl()
}

///|
pub fn WorkerService::restart(
  self : WorkerService,
) -> WorkerServiceRestart raise WorkerServiceError {
  self.require_raw_mode()
  self.require_no_maintenance()
  self.restart_impl()
}

///|
pub fn WorkerService::retry_restart_cleanup(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_raw_mode()
  self.require_no_maintenance()
  self.retry_restart_cleanup_impl()
}

///|
pub fn WorkerService::shutdown_clean(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_raw_mode()
  self.require_no_maintenance()
  self.shutdown_clean_impl()
}

///|
pub fn WorkerService::retry_close(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_raw_mode()
  self.require_no_maintenance()
  self.retry_close_impl()
}

///|
pub fn WorkerService::close(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_raw_mode()
  self.require_no_maintenance()
  self.close_impl()
}

///|
/// Starts cooperative child retirement while retaining every scheduler A/B
/// obligation in this owner.
pub fn WorkerService::begin_recovery_maintenance(
  self : WorkerService,
  failure : @worker_protocol.WorkerFailure,
) -> Unit raise WorkerServiceError {
  self.require_raw_mode()
  self.begin_recovery_maintenance_impl(failure)
}

///|
/// Starts cooperative clean shutdown after exact quiescence preflight.
pub fn WorkerService::begin_shutdown_maintenance(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_raw_mode()
  self.begin_shutdown_maintenance_impl()
}

///|
/// Starts cooperative cleanup of the exact failed replacement child.
pub fn WorkerService::begin_restart_cleanup_maintenance(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_raw_mode()
  self.begin_restart_cleanup_maintenance_impl()
}

///|
/// Advances at most one process/root transition or scheduler obligation.
pub fn WorkerService::progress_maintenance(
  self : WorkerService,
) -> LunaWorkerServiceMaintenanceProgress raise WorkerServiceError {
  self.require_raw_mode()
  self.progress_maintenance_impl()
}

///|
/// Returns the current child-maintenance wake bound without waiting.
pub fn WorkerService::maintenance_remaining_millis(
  self : WorkerService,
) -> Int raise WorkerServiceError {
  self.require_raw_mode()
  self.maintenance_remaining_millis_impl()
}
