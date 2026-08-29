///|
fn RootBoundWorkerProcessSupervisor::begin_child_maintenance(
  self : RootBoundWorkerProcessSupervisor,
  mode : Int,
) -> Unit raise RootBoundWorkerProcessError {
  if self.maintenance_mode > 0 || self.failed_replacement is Some(_) {
    raise RootBoundInvalidLifecycle
  }
  self.clear_exchange()
  self.process.begin_shutdown_maintenance() catch {
    error => raise RootBoundFailure(Process(error))
  }
  self.maintenance_mode = mode
}

///|
/// Begins recovery cleanup while retaining the exact root pair for restart.
pub fn RootBoundWorkerProcessSupervisor::begin_recovery_maintenance(
  self : RootBoundWorkerProcessSupervisor,
) -> Unit raise RootBoundWorkerProcessError {
  if self.lifecycle != RootBoundReady &&
    self.lifecycle != RootBoundRecoveryRequired &&
    self.lifecycle != RootBoundRecoveryClosingFailed {
    raise RootBoundInvalidLifecycle
  }
  self.begin_child_maintenance(1)
  self.lifecycle = RootBoundRecoveryRequired
}

///|
/// Begins terminal cleanup. Roots remain live until child reap and native
/// handle close have both completed.
pub fn RootBoundWorkerProcessSupervisor::begin_close_maintenance(
  self : RootBoundWorkerProcessSupervisor,
) -> Unit raise RootBoundWorkerProcessError {
  if self.lifecycle == RootBoundClosed ||
    self.maintenance_mode > 0 ||
    self.failed_replacement is Some(_) {
    raise RootBoundInvalidLifecycle
  }
  if self.process.lifecycle() == Closed {
    self.clear_exchange()
    self.maintenance_mode = 3
  } else {
    self.begin_child_maintenance(2)
  }
  self.lifecycle = RootBoundClosingFailed
}

///|
/// Begins cooperative cleanup of the exact failed replacement child. The
/// retained roots and predecessor remain live for a later retry of restart.
pub fn RootBoundWorkerProcessSupervisor::begin_replacement_cleanup_maintenance(
  self : RootBoundWorkerProcessSupervisor,
) -> Unit raise RootBoundWorkerProcessError {
  if self.lifecycle != RootBoundReplacementCleanup || self.maintenance_mode > 0 {
    raise RootBoundInvalidLifecycle
  }
  guard self.failed_replacement is Some(failed) else {
    raise RootBoundInvalidLifecycle
  }
  failed.begin_cleanup_maintenance() catch {
    error => raise RootBoundFailure(Process(error))
  }
  self.maintenance_mode = 4
}

///|
fn RootBoundWorkerProcessSupervisor::record_child_maintenance_failure(
  self : RootBoundWorkerProcessSupervisor,
) -> Unit {
  self.child_cleanup_failure = match self.process.maintenance_failure {
    Some(error) => Some(Process(error))
    None => Some(Process(Transport(OperationFailed)))
  }
}

///|
fn RootBoundWorkerProcessSupervisor::apply_retained_child_progress(
  self : RootBoundWorkerProcessSupervisor,
  progress : WorkerProcessMaintenanceProgress,
) -> WorkerProcessMaintenanceProgress {
  if progress.is_cleanup_required() {
    self.record_child_maintenance_failure()
    return worker_maintenance_cleanup_required
  }
  if progress.is_cleanup_stuck() {
    return worker_maintenance_cleanup_stuck
  }
  if !progress.is_complete() {
    if progress.is_advanced() {
      return worker_maintenance_advanced
    }
    return worker_maintenance_pending
  }
  if self.maintenance_mode == 1 {
    self.maintenance_mode = -1
    self.lifecycle = RootBoundRecovering
    return worker_maintenance_complete
  }
  self.maintenance_mode = 3
  worker_maintenance_advanced
}

///|
/// Advances one child transition, or after complete child reap, one root-pair
/// close. A single call never does both.
pub fn RootBoundWorkerProcessSupervisor::progress_maintenance(
  self : RootBoundWorkerProcessSupervisor,
) -> WorkerProcessMaintenanceProgress raise RootBoundWorkerProcessError {
  if self.maintenance_mode == -1 || self.maintenance_mode == -2 {
    return worker_maintenance_complete
  }
  if self.maintenance_mode == 4 {
    guard self.failed_replacement is Some(failed) else {
      raise RootBoundInvalidLifecycle
    }
    let progress = failed.progress_cleanup_maintenance() catch {
      error => {
        self.child_cleanup_failure = Some(Process(error))
        raise RootBoundFailure(Process(error))
      }
    }
    if progress.is_cleanup_required() {
      self.child_cleanup_failure = Some(Process(failed.cleanup_failure()))
      return worker_maintenance_cleanup_required
    }
    if progress.is_cleanup_stuck() {
      return worker_maintenance_cleanup_stuck
    }
    if !progress.is_complete() {
      if progress.is_advanced() {
        return worker_maintenance_advanced
      }
      return worker_maintenance_pending
    }
    self.failed_replacement = None
    self.child_cleanup_failure = None
    self.maintenance_mode = -1
    self.lifecycle = RootBoundRecovering
    return worker_maintenance_complete
  }
  if self.maintenance_mode == 1 || self.maintenance_mode == 2 {
    let progress = self.process.progress_shutdown_maintenance() catch {
      error => {
        self.child_cleanup_failure = Some(Process(error))
        raise RootBoundFailure(Process(error))
      }
    }
    return self.apply_retained_child_progress(progress)
  }
  if self.maintenance_mode != 3 {
    raise RootBoundInvalidLifecycle
  }
  let status = close_root_pair_status(self.require_roots())
  let roots_live = roots_still_live(self.require_roots())
  self.apply_root_close_progress(status, roots_live)
}

///|
fn RootBoundWorkerProcessSupervisor::apply_root_close_progress(
  self : RootBoundWorkerProcessSupervisor,
  status : Int,
  roots_live : Bool,
) -> WorkerProcessMaintenanceProgress {
  self.roots_cleanup_failure = root_failure_status(status)
  if roots_live {
    self.lifecycle = RootBoundClosingFailed
    return worker_maintenance_cleanup_required
  }
  self.roots_selected = false
  self.lifecycle = RootBoundClosed
  self.maintenance_mode = -2
  if status != 0 {
    worker_maintenance_cleanup_required
  } else {
    worker_maintenance_complete
  }
}

///|
pub fn RootBoundWorkerProcessSupervisor::maintenance_remaining_millis(
  self : RootBoundWorkerProcessSupervisor,
) -> Int raise RootBoundWorkerProcessError {
  if self.maintenance_mode == 4 {
    guard self.failed_replacement is Some(failed) else {
      raise RootBoundInvalidLifecycle
    }
    return failed.cleanup_maintenance_remaining_millis() catch {
      error => raise RootBoundFailure(Process(error))
    }
  }
  if self.maintenance_mode != 1 && self.maintenance_mode != 2 {
    return 0
  }
  self.process.shutdown_maintenance_remaining_millis() catch {
    error => raise RootBoundFailure(Process(error))
  }
}

///|
pub fn RootBoundWorkerProcessSupervisor::maintenance_exit(
  self : RootBoundWorkerProcessSupervisor,
) -> WorkerProcessExit raise RootBoundWorkerProcessError {
  self.process.shutdown_maintenance_exit() catch {
    error => raise RootBoundFailure(Process(error))
  }
}
