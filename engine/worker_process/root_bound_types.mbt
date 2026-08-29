///|
pub(all) enum RootBoundWorkerProcessLifecycle {
  RootBoundReady
  RootBoundRecoveryRequired
  RootBoundRecoveryClosingFailed
  RootBoundRecovering
  RootBoundReplacementCleanup
  RootBoundClosingFailed
  RootBoundClosed
} derive(Debug, Eq)

///|
pub(all) enum RootBoundWorkerProcessFailure {
  Process(WorkerProcessError)
  ModelRootBinding(@approved_fs.ApprovedFsError)
  KernelRootBinding(@approved_fs.ApprovedFsError)
  RootsBusy
  RootsCloseFailed
  ExecutableBusy
  ExecutableCloseFailed
} derive(Debug, Eq)

///|
pub(all) suberror RootBoundWorkerProcessError {
  RootBoundInvalidLifecycle
  RootBoundOwnerMismatch
  RootBoundFailure(RootBoundWorkerProcessFailure)
  RootBoundCompoundFailure(
    RootBoundWorkerProcessFailure,
    RootBoundWorkerProcessFailure
  )
  RootBoundTripleFailure(
    RootBoundWorkerProcessFailure,
    RootBoundWorkerProcessFailure,
    RootBoundWorkerProcessFailure
  )
} derive(Debug, Eq)

///|
/// Instance-lifetime owner of one exact executable, limits, canonical source,
/// child sequence domain, and opaque model/kernel root lease pair.
pub struct RootBoundWorkerProcessSupervisor {
  priv approved_executable : @worker_executable_file.WorkerExecutableAdmission
  priv limits : WorkerProcessLimits
  priv roots : @approved_fs.WorkerApprovedRoots
  priv mut roots_selected : Bool
  priv mut process : WorkerProcessSupervisor
  priv mut failed_replacement : FailedWorkerProcessStartup?
  priv mut cause : RootBoundWorkerProcessFailure?
  priv mut child_cleanup_failure : RootBoundWorkerProcessFailure?
  priv mut roots_cleanup_failure : RootBoundWorkerProcessFailure?
  priv mut transport_mode : Int
  priv mut exchange_epoch : UInt64
  priv mut active_exchange_epoch : UInt64
  priv mut exchange_phase : Int
  priv mut exchange_side : Int
  priv mut exchange_sequence : UInt64
  priv exchange_plan : @worker_protocol.SubmittedSchedulePlanSlot
  priv mut maintenance_mode : Int
  priv mut lifecycle : RootBoundWorkerProcessLifecycle
}

///|
/// Cleanup-only result when initial startup did not publish a supervisor.
pub struct FailedRootBoundWorkerProcessStartup {
  priv approved_executable : @worker_executable_file.WorkerExecutableAdmission
  priv roots : @approved_fs.WorkerApprovedRoots
  priv mut roots_selected : Bool
  priv mut child : FailedWorkerProcessStartup?
  priv mut child_cleanup_process : WorkerProcessError?
  priv mut child_cleanup_native : @process.ProcessError?
  priv mut roots_cleanup_status : Int
  priv mut closed : Bool
}

///|
#valtype
pub(all) enum RootBoundWorkerProcessPreparation {
  ReadyRootBoundProcess(RootBoundWorkerProcessSupervisor)
  RootBoundCleanupRequired(FailedRootBoundWorkerProcessStartup)
}

///|
pub(all) enum RootBoundWorkerProcessRestart {
  RootBoundRestarted
  RootBoundRestartCleanupRequired
}

///|
#valtype
pub struct RootBoundWorkerSubmission(
  RootBoundWorkerProcessSupervisor,
  Int,
  UInt64
)

///|
#valtype
pub struct RootBoundReceivedWorkerCompletion(
  RootBoundWorkerProcessSupervisor,
  Int,
  UInt64
)

///|
#valtype
pub enum RootBoundWorkerExchangeProgress {
  ExchangePending
  ExchangeCompleted
}

///|
pub fn RootBoundWorkerProcessSupervisor::lifecycle(
  self : RootBoundWorkerProcessSupervisor,
) -> RootBoundWorkerProcessLifecycle {
  self.lifecycle
}

///|
pub fn RootBoundWorkerProcessSupervisor::cause(
  self : RootBoundWorkerProcessSupervisor,
) -> RootBoundWorkerProcessFailure? {
  self.cause
}

///|
pub fn RootBoundWorkerProcessSupervisor::child_cleanup_failure(
  self : RootBoundWorkerProcessSupervisor,
) -> RootBoundWorkerProcessFailure? {
  self.child_cleanup_failure
}

///|
pub fn RootBoundWorkerProcessSupervisor::roots_cleanup_failure(
  self : RootBoundWorkerProcessSupervisor,
) -> RootBoundWorkerProcessFailure? {
  self.roots_cleanup_failure
}

///|
pub fn RootBoundWorkerProcessSupervisor::is_closed(
  self : RootBoundWorkerProcessSupervisor,
) -> Bool {
  self.lifecycle == RootBoundClosed
}

///|
pub fn FailedRootBoundWorkerProcessStartup::cause(
  self : FailedRootBoundWorkerProcessStartup,
) -> RootBoundWorkerProcessFailure {
  match self.child {
    Some(child) => Process(child.cause())
    None => Process(Invalid(Lifecycle))
  }
}

///|
pub fn FailedRootBoundWorkerProcessStartup::child_cleanup_failure(
  self : FailedRootBoundWorkerProcessStartup,
) -> RootBoundWorkerProcessFailure? {
  match self.child_cleanup_process {
    Some(error) => Some(Process(error))
    None =>
      match self.child {
        Some(child) if child.cleanup_status != 0 =>
          Some(Process(child.cleanup_failure()))
        _ =>
          match self.child_cleanup_native {
            Some(error) => Some(Process(transport_error(error)))
            None => None
          }
      }
  }
}

///|
pub fn FailedRootBoundWorkerProcessStartup::roots_cleanup_failure(
  self : FailedRootBoundWorkerProcessStartup,
) -> RootBoundWorkerProcessFailure? {
  root_failure_status(self.roots_cleanup_status)
}

///|
pub fn FailedRootBoundWorkerProcessStartup::is_closed(
  self : FailedRootBoundWorkerProcessStartup,
) -> Bool {
  self.closed
}
