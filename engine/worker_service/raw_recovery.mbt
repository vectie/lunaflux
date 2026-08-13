///|
pub fn WorkerService::begin_recovery(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_raw_mode()
  self.begin_recovery_impl()
}

///|
pub fn WorkerService::recover_flight(
  self : WorkerService,
  failure : @worker_protocol.WorkerFailure,
) -> UInt64 raise WorkerServiceError {
  self.require_raw_mode()
  self.recover_flight_impl(failure)
}

///|
pub fn WorkerService::invalidate_device_state(
  self : WorkerService,
) -> Int raise WorkerServiceError {
  self.require_raw_mode()
  self.invalidate_device_state_impl()
}

///|
pub fn WorkerService::drain_restart_forbidden(
  self : WorkerService,
) -> @core.InstanceLossDrain raise WorkerServiceError {
  self.require_raw_mode()
  self.drain_restart_forbidden_impl()
}

///|
pub fn WorkerService::abandon_restart(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_raw_mode()
  self.abandon_restart_impl()
}

///|
pub fn WorkerService::restart(
  self : WorkerService,
) -> WorkerServiceRestart raise WorkerServiceError {
  self.require_raw_mode()
  self.restart_impl()
}

///|
pub fn WorkerService::retry_restart_cleanup(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_raw_mode()
  self.retry_restart_cleanup_impl()
}

///|
pub fn WorkerService::shutdown_clean(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_raw_mode()
  self.shutdown_clean_impl()
}

///|
pub fn WorkerService::retry_close(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_raw_mode()
  self.retry_close_impl()
}

///|
pub fn WorkerService::close(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  self.require_raw_mode()
  self.close_impl()
}
