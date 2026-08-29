///|
fn root_failure(
  error : @approved_fs.ApprovedFsError,
) -> RootBoundWorkerProcessFailure {
  match error {
    Busy(_) => RootsBusy
    _ => RootsCloseFailed
  }
}

///|
fn close_root_pair(
  roots : @approved_fs.WorkerApprovedRoots,
) -> RootBoundWorkerProcessFailure? {
  try @approved_fs.close_worker_approved_roots(roots) catch {
    error => Some(root_failure(error))
  } noraise {
    _ => None
  }
}

///|
fn close_root_pair_status(roots : @approved_fs.WorkerApprovedRoots) -> Int {
  @approved_fs.worker_approved_roots_close_status(roots)
}

///|
fn root_failure_status(status : Int) -> RootBoundWorkerProcessFailure? {
  if status == 0 {
    None
  } else if status == 7 {
    Some(RootsBusy)
  } else {
    Some(RootsCloseFailed)
  }
}

///|
fn close_approved_executable(
  executable : @worker_executable_file.WorkerExecutableAdmission,
) -> RootBoundWorkerProcessFailure? {
  try executable.close() catch {
    FileBusy => Some(ExecutableBusy)
    _ => Some(ExecutableCloseFailed)
  } noraise {
    _ => None
  }
}

///|
fn approved_executable_live(
  executable : @worker_executable_file.WorkerExecutableAdmission,
) -> Bool {
  executable.is_activation_live()
}

///|
fn roots_still_live(roots : @approved_fs.WorkerApprovedRoots) -> Bool {
  !@approved_fs.worker_approved_roots_is_closed(roots)
}

///|
fn RootBoundWorkerProcessSupervisor::require_roots(
  self : RootBoundWorkerProcessSupervisor,
) -> @approved_fs.WorkerApprovedRoots {
  if !self.roots_selected {
    abort("root-bound roots selection invariant")
  }
  self.roots
}
