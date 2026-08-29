///|
let idle_child_alive : ChildProcessIdleObservation = {
  kind: 0,
  exit_kind: 0,
  exit_code: 0,
}

///|
fn idle_child_exited(kind : Int, code : Int) -> ChildProcessIdleObservation {
  { kind: 1, exit_kind: kind, exit_code: code }
}

///|
/// Performs exactly one nonblocking child-liveness transition.
///
/// The caller must serialize this observation with frame I/O and maintenance.
/// A live result changes no owner state. An exited result records the exact
/// reaped status in this owner and moves it directly to the existing
/// reaped-descriptor-close phase; subsequent shutdown-maintenance progress
/// closes the descriptor without signaling or waiting for the child again.
pub fn ChildProcess::observe_idle_exit(
  self : ChildProcess,
) -> ChildProcessIdleObservation raise ProcessError {
  if self.closed ||
    self.maintenance_phase != 0 ||
    self.write_transfer_active ||
    self.read_transfer_active {
    raise InvalidState(Wait)
  }
  let result = raw_try_wait(self.handle)
  let status = maintenance_wait_status(result)
  if status == 5 {
    return idle_child_alive
  }
  if status != 0 {
    self.failed = true
    raise OperationFailed(Wait)
  }
  let kind = maintenance_wait_kind(result)
  let code = maintenance_wait_code(result)
  self.failed = true
  self.record_maintenance_exit(kind, code)
  idle_child_exited(kind, code)
}
