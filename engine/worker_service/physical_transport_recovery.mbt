///|
/// Private physical-worker family. Scheduler, request, KV, and public online
/// owners never branch on this value. A tensor-parallel variant may be added
/// only when its rooted owner provides the same recovery, replacement, and
/// terminal-cleanup contract as the rooted single-worker owner.
#valtype
priv enum WorkerPhysicalTransport {
  RemoteGroup(RemoteWorkerServiceControl)
  Uninstalled
  RootedSingle(@worker_process.RootBoundWorkerProcessSupervisor)
  TensorParallelGroup(
    @tensor_parallel_group_transport.TensorParallelGroupTransport
  )
}

///|
#valtype
priv enum WorkerPhysicalExchangeProgress {
  PhysicalExchangePending
  PhysicalExchangeCompleted
}

///|
priv struct WorkerPhysicalMaintenanceProgress(Int32)

///|
#valtype
priv enum WorkerPhysicalRestart {
  PhysicalRestarted
  PhysicalRestartCleanupRequired
}

///|
fn WorkerPhysicalTransport::claim(
  self : WorkerPhysicalTransport,
  predecessor : UInt64,
) -> Unit raise WorkerServiceError {
  match self {
    RemoteGroup(group) =>
      if group.previous != predecessor {
        raise Invalid(Sequence)
      }
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) =>
      process.claim_exchange(predecessor) catch {
        error => raise physical_transport_error(error)
      }
    TensorParallelGroup(group) =>
      if group.previous_plan_sequence_value() != predecessor {
        raise Invalid(Sequence)
      }
  }
}

///|
fn WorkerPhysicalTransport::is_recovering(
  self : WorkerPhysicalTransport,
) -> Bool {
  match self {
    RemoteGroup(group) => group.recovering
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) => process.lifecycle() == RootBoundRecovering
    TensorParallelGroup(group) => group.has_recovery_obligation()
  }
}

///|
/// True only after the failed physical owner has completed deterministic
/// child cleanup and retained the exact replacement obligation. A TP group
/// remains `Recovering` while any rank child is still live, so callers must
/// not retire the scheduler flight merely because recovery has begun.
fn WorkerPhysicalTransport::recovery_cleanup_complete(
  self : WorkerPhysicalTransport,
) -> Bool {
  match self {
    RemoteGroup(group) => group.current.recovery_cleanup_complete()
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    // RootBound recovery is published only after its single child is closed.
    RootedSingle(process) => process.lifecycle() == RootBoundRecovering
    TensorParallelGroup(group) =>
      group.lifecycle() == Prepared && group.has_recovery_obligation()
  }
}

///|
/// Selects the bounded service failure that belongs to the exact physical cut.
/// A TP report, once published, takes precedence over ambient compatibility
/// input; the single-worker path preserves its established caller vocabulary.
fn WorkerPhysicalTransport::recovery_failure(
  self : WorkerPhysicalTransport,
  fallback : @worker_protocol.WorkerFailure,
) -> @worker_protocol.WorkerFailure raise WorkerServiceError {
  match self {
    RemoteGroup(group) => group.failure()
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(_) => fallback
    TensorParallelGroup(group) =>
      if group.failure_report_invalid() {
        ExecutionFailed
      } else {
        tensor_parallel_worker_failure(
          group.taken_group_failure() catch {
            error => raise tensor_parallel_transport_error(error)
          },
        )
      }
  }
}

///|
fn WorkerPhysicalTransport::allows_begin_recovery(
  self : WorkerPhysicalTransport,
) -> Bool {
  match self {
    RemoteGroup(group) => group.current.lifecycle() == Faulted
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) =>
      match process.lifecycle() {
        RootBoundReady
        | RootBoundRecoveryRequired
        | RootBoundRecoveryClosingFailed => true
        _ => false
      }
    TensorParallelGroup(group) => group.lifecycle() == Faulted
  }
}

///|
fn WorkerPhysicalTransport::has_exchange_capacity(
  self : WorkerPhysicalTransport,
) -> Bool {
  match self {
    RemoteGroup(group) =>
      !group.recovering && group.current.lifecycle() == Ready
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) => process.has_exchange_capacity()
    TensorParallelGroup(group) => group.has_exchange_capacity()
  }
}

///|
fn WorkerPhysicalTransport::probe_idle_health(
  self : WorkerPhysicalTransport,
) -> Unit raise WorkerServiceError {
  match self {
    RemoteGroup(group) =>
      group.current.probe_idle_health() catch {
        error => raise remote_error(error)
      }
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(_) => ()
    TensorParallelGroup(group) =>
      group.probe_idle_rank() catch {
        error => raise tensor_parallel_transport_error(error)
      }
  }
}
