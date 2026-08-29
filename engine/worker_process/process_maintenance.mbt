///|
let worker_maintenance_pending : WorkerProcessMaintenanceProgress = { kind: 0 }

///|
let worker_maintenance_advanced : WorkerProcessMaintenanceProgress = { kind: 1 }

///|
let worker_maintenance_complete : WorkerProcessMaintenanceProgress = { kind: 2 }

///|
let worker_maintenance_cleanup_required : WorkerProcessMaintenanceProgress = {
  kind: 3,
}

///|
let worker_maintenance_cleanup_stuck : WorkerProcessMaintenanceProgress = {
  kind: 4,
}

///|
/// Revokes transport progress and begins cleanup in the exact retained child.
/// Submitted/received A/B sides remain only as scheduler recovery obligations.
pub fn WorkerProcessSupervisor::begin_shutdown_maintenance(
  self : WorkerProcessSupervisor,
) -> Unit raise WorkerProcessError {
  if self.maintenance_phase != 0 || self.lifecycle == Closed {
    raise Invalid(Lifecycle)
  }
  self.require_child().begin_shutdown_maintenance() catch {
    error => raise transport_error(error)
  }
  self.maintenance_failure = None
  self.maintenance_phase = 1
  self.lifecycle = ClosingFailed
}

///|
/// Delegates at most one cooperative native-process transition.
pub fn WorkerProcessSupervisor::progress_shutdown_maintenance(
  self : WorkerProcessSupervisor,
) -> WorkerProcessMaintenanceProgress raise WorkerProcessError {
  if self.maintenance_phase == 0 {
    raise Invalid(Lifecycle)
  }
  if self.maintenance_phase == 2 {
    return worker_maintenance_complete
  }
  let progress = self.require_child().progress_shutdown_maintenance() catch {
      error => {
        let mapped = transport_error(error)
        self.maintenance_failure = Some(mapped)
        self.lifecycle = ClosingFailed
        raise mapped
      }
    }
  self.apply_maintenance_progress(
    progress.is_cleanup_required(),
    progress.is_cleanup_stuck(),
    progress.is_complete(),
    progress.is_advanced(),
  )
}

///|
fn WorkerProcessSupervisor::apply_maintenance_progress(
  self : WorkerProcessSupervisor,
  cleanup_required : Bool,
  cleanup_stuck : Bool,
  complete : Bool,
  advanced : Bool,
) -> WorkerProcessMaintenanceProgress {
  if cleanup_required {
    self.maintenance_failure = Some(Transport(OperationFailed))
    self.lifecycle = ClosingFailed
    return worker_maintenance_cleanup_required
  }
  if cleanup_stuck {
    self.lifecycle = ClosingFailed
    return worker_maintenance_cleanup_stuck
  }
  if complete {
    self.maintenance_phase = 2
    self.lifecycle = Closed
    return worker_maintenance_complete
  }
  if advanced {
    worker_maintenance_advanced
  } else {
    worker_maintenance_pending
  }
}

///|
pub fn WorkerProcessSupervisor::shutdown_maintenance_remaining_millis(
  self : WorkerProcessSupervisor,
) -> Int raise WorkerProcessError {
  if self.maintenance_phase == 0 {
    raise Invalid(Lifecycle)
  }
  self.require_child().maintenance_remaining_millis() catch {
    error => raise transport_error(error)
  }
}

///|
pub fn WorkerProcessSupervisor::shutdown_maintenance_exit(
  self : WorkerProcessSupervisor,
) -> WorkerProcessExit raise WorkerProcessError {
  if self.maintenance_phase == 0 {
    raise Invalid(Lifecycle)
  }
  let exit = self.require_child().maintenance_exit() catch {
      error => raise transport_error(error)
    }
  { kind: exit.kind(), code: exit.code() }
}
