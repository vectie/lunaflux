///|
/// Return or validate one exact admitted policy field.
pub fn ServicePolicyClaims::listen_host(self : ServicePolicyClaims) -> String {
  self.listen_host
}

///|
/// Return or validate one exact admitted policy field.
pub fn ServicePolicyClaims::listen_port(self : ServicePolicyClaims) -> Int {
  self.listen_port
}

///|
/// Return or validate one exact admitted policy field.
pub fn ServicePolicyClaims::max_request_bytes(
  self : ServicePolicyClaims,
) -> Int {
  self.max_request_bytes
}

///|
/// Return or validate one exact admitted policy field.
pub fn ServicePolicyClaims::graceful_drain_milliseconds(
  self : ServicePolicyClaims,
) -> Int {
  self.graceful_drain_milliseconds
}

///|
/// Return or validate one exact admitted policy field.
pub fn ServicePolicyClaims::diagnostic_mode(
  self : ServicePolicyClaims,
) -> String {
  self.diagnostic_mode
}

///|
/// Return or validate one exact admitted policy field.
pub fn WorkerProcessPolicyClaims::max_frame_bytes(
  self : WorkerProcessPolicyClaims,
) -> Int {
  self.max_frame_bytes
}

///|
/// Return or validate one exact admitted policy field.
pub fn WorkerProcessPolicyClaims::startup_io_timeout_millis(
  self : WorkerProcessPolicyClaims,
) -> Int {
  self.startup_io_timeout_millis
}

///|
/// Return or validate one exact admitted policy field.
pub fn WorkerProcessPolicyClaims::io_timeout_millis(
  self : WorkerProcessPolicyClaims,
) -> Int {
  self.io_timeout_millis
}

///|
/// Return or validate one exact admitted policy field.
pub fn WorkerProcessPolicyClaims::shutdown_timeout_millis(
  self : WorkerProcessPolicyClaims,
) -> Int {
  self.shutdown_timeout_millis
}

///|
/// Return or validate one exact admitted policy field.
pub fn RestartPolicyClaims::initial_backoff_millis(
  self : RestartPolicyClaims,
) -> UInt64 {
  self.initial_backoff_millis
}

///|
/// Return or validate one exact admitted policy field.
pub fn RestartPolicyClaims::maximum_backoff_millis(
  self : RestartPolicyClaims,
) -> UInt64 {
  self.maximum_backoff_millis
}

///|
/// Return or validate one exact admitted policy field.
pub fn RestartPolicyClaims::stable_after_millis(
  self : RestartPolicyClaims,
) -> UInt64 {
  self.stable_after_millis
}

///|
/// Return or validate one exact admitted policy field.
pub fn RestartPolicyClaims::maximum_attempts(self : RestartPolicyClaims) -> Int {
  self.maximum_attempts
}

///|
/// Return or validate one exact admitted policy field.
pub fn TransportPolicyClaims::read_chunk_bytes(
  self : TransportPolicyClaims,
) -> Int {
  self.read_chunk_bytes
}

///|
/// Return or validate one exact admitted policy field.
pub fn TransportPolicyClaims::write_chunk_bytes(
  self : TransportPolicyClaims,
) -> Int {
  self.write_chunk_bytes
}

///|
/// Return or validate one exact admitted policy field.
pub fn TransportPolicyClaims::accept_timeout_millis(
  self : TransportPolicyClaims,
) -> Int {
  self.accept_timeout_millis
}

///|
/// Return or validate one exact admitted policy field.
pub fn TransportPolicyClaims::input_idle_timeout_millis(
  self : TransportPolicyClaims,
) -> Int {
  self.input_idle_timeout_millis
}

///|
/// Return or validate one exact admitted policy field.
pub fn TransportPolicyClaims::write_timeout_millis(
  self : TransportPolicyClaims,
) -> Int {
  self.write_timeout_millis
}

///|
/// Return or validate one exact admitted policy field.
pub fn TransportPolicyClaims::reactor_transition_budget(
  self : TransportPolicyClaims,
) -> Int {
  self.reactor_transition_budget
}

///|
/// Return or validate one exact admitted policy field.
pub fn PreparationPolicyClaims::framed_max_frame_bytes(
  self : PreparationPolicyClaims,
) -> Int {
  self.framed_max_frame_bytes
}

///|
/// Return or validate one exact admitted policy field.
pub fn PreparationPolicyClaims::lane_count(
  self : PreparationPolicyClaims,
) -> Int {
  self.lane_count
}

///|
/// Return or validate one exact admitted policy field.
pub fn PreparationPolicyClaims::step_work_units(
  self : PreparationPolicyClaims,
) -> Int {
  self.step_work_units
}

///|
/// Return or validate one exact admitted policy field.
pub fn PreparationPolicyClaims::total_work_units(
  self : PreparationPolicyClaims,
) -> UInt64 {
  self.total_work_units
}

///|
/// Return or validate one exact admitted policy field.
pub fn PreparationPolicyClaims::storage_int_cells(
  self : PreparationPolicyClaims,
) -> UInt64 {
  self.storage_int_cells
}

///|
/// Return or validate one exact admitted policy field.
pub fn PreparationPolicyClaims::storage_byte_cells(
  self : PreparationPolicyClaims,
) -> UInt64 {
  self.storage_byte_cells
}

///|
/// Return or validate one exact admitted policy field.
pub fn PreparationPolicyClaims::storage_reference_cells(
  self : PreparationPolicyClaims,
) -> UInt64 {
  self.storage_reference_cells
}

///|
/// Return or validate one exact admitted policy field.
pub fn PreparationPolicyClaims::event_work_units(
  self : PreparationPolicyClaims,
) -> Int {
  self.event_work_units
}

///|
/// Return or validate one exact admitted policy field.
pub fn TelemetryPolicyClaims::instance_log_capacity(
  self : TelemetryPolicyClaims,
) -> Int {
  self.instance_log_capacity
}
