///|
fn physical_transport_failure(
  error : @worker_process.RootBoundWorkerProcessError,
) -> WorkerPhysicalTransportFailure {
  match error {
    RootBoundInvalidLifecycle | RootBoundOwnerMismatch =>
      PhysicalLifecycleViolation
    RootBoundFailure(Process(Backpressured)) => PhysicalBackpressured
    RootBoundFailure(
      Process(Wire(_) | Protocol(_) | Invalid(Completion | Sequence))
    ) => PhysicalInvalidOutput
    RootBoundCompoundFailure(_, _) | RootBoundTripleFailure(_, _, _) =>
      PhysicalCleanupFailed
    RootBoundFailure(RootsBusy | RootsCloseFailed) => PhysicalCleanupFailed
    _ => PhysicalUnavailable
  }
}

///|
fn physical_transport_error(
  error : @worker_process.RootBoundWorkerProcessError,
) -> WorkerServiceError {
  PhysicalTransport(physical_transport_failure(error))
}

///|
fn physical_cleanup_failure(
  error : @worker_process.RootBoundWorkerProcessError,
) -> WorkerPhysicalTransportFailure {
  match error {
    RootBoundInvalidLifecycle | RootBoundOwnerMismatch =>
      PhysicalLifecycleViolation
    _ => PhysicalCleanupFailed
  }
}

///|
fn physical_cleanup_error(
  error : @worker_process.RootBoundWorkerProcessError,
) -> WorkerServiceError {
  PhysicalTransport(physical_cleanup_failure(error))
}

///|
fn rooted_single_transport(
  process : @worker_process.RootBoundWorkerProcessSupervisor,
) -> WorkerPhysicalTransport {
  RootedSingle(process)
}

///|
fn tensor_parallel_group_transport(
  group : @tensor_parallel_group_transport.TensorParallelGroupTransport,
) -> WorkerPhysicalTransport {
  TensorParallelGroup(group)
}

///|
fn tensor_parallel_transport_failure(
  error : @tensor_parallel_group_transport.TensorParallelGroupTransportError,
) -> WorkerPhysicalTransportFailure {
  match error {
    Rejected(rule~, rank=_) =>
      match rule {
        Completion => PhysicalInvalidOutput
        Cleanup | Root => PhysicalCleanupFailed
        Lifecycle | SchedulerPredecessor => PhysicalLifecycleViolation
        _ => PhysicalUnavailable
      }
  }
}

///|
fn tensor_parallel_transport_error(
  error : @tensor_parallel_group_transport.TensorParallelGroupTransportError,
) -> WorkerServiceError {
  PhysicalTransport(tensor_parallel_transport_failure(error))
}

///|
/// Sanitizes the exact authenticated group report into the public worker-row
/// vocabulary. Rank identity and protocol details never cross the service
/// boundary.
fn tensor_parallel_worker_failure(
  report : @rank_group_protocol.RankGroupFailureReport,
) -> @worker_protocol.WorkerFailure {
  tensor_parallel_worker_failure_reason(report.reason())
}

///|
fn tensor_parallel_worker_failure_reason(
  reason : @rank_group_protocol.RankGroupFailureReason,
) -> @worker_protocol.WorkerFailure {
  match reason {
    ProtocolViolation(_) | RankExecutionFailure => ExecutionFailed
    DeadlineExpired | RankLost | CollectiveFailure | HostFailure =>
      ResourceUnavailable
  }
}

///|
fn take_tensor_parallel_worker_failure(
  group : @tensor_parallel_group_transport.TensorParallelGroupTransport,
) -> @worker_protocol.WorkerFailure raise WorkerServiceError {
  try group.take_group_failure() catch {
    error =>
      if group.failure_report_invalid() {
        ExecutionFailed
      } else {
        raise tensor_parallel_transport_error(error)
      }
  } noraise {
    report => tensor_parallel_worker_failure(report)
  }
}
