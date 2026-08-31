///|
/// Private physical-worker family. Scheduler, request, KV, and public online
/// owners never branch on this value. A tensor-parallel variant may be added
/// only when its rooted owner provides the same recovery, replacement, and
/// terminal-cleanup contract as the rooted single-worker owner.
#valtype
priv enum WorkerPhysicalTransport {
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
priv struct WorkerPhysicalMaintenanceProgress(Int)

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
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(_) => ()
    TensorParallelGroup(group) =>
      group.probe_idle_rank() catch {
        error => raise tensor_parallel_transport_error(error)
      }
  }
}

///|
fn WorkerPhysicalTransport::begin_exchange(
  self : WorkerPhysicalTransport,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> UInt64 raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) =>
      process.begin_exchange(plan) catch {
        error => raise physical_transport_error(error)
      }
    TensorParallelGroup(group) => {
      let sequence = plan.sequence_wire() catch {
        error => raise Scheduler(WorkerProtocol(error))
      }
      group.begin(plan) catch {
        error => raise tensor_parallel_transport_error(error)
      }
      sequence
    }
  }
}

///|
fn WorkerPhysicalTransport::has_submission(
  self : WorkerPhysicalTransport,
  sequence : UInt64,
) -> Bool raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) =>
      try process.submission_for_sequence(sequence) catch {
        RootBoundFailure(Process(Invalid(Submission))) => false
        error => raise physical_transport_error(error)
      } noraise {
        _ => true
      }
    TensorParallelGroup(group) => group.has_submission(sequence)
  }
}

///|
fn WorkerPhysicalTransport::progress_exchange(
  self : WorkerPhysicalTransport,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> WorkerPhysicalExchangeProgress raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) => {
      let progress = process.progress_exchange() catch {
        error => raise physical_transport_error(error)
      }
      match progress {
        ExchangePending => PhysicalExchangePending
        ExchangeCompleted => PhysicalExchangeCompleted
      }
    }
    TensorParallelGroup(group) => {
      let progress = group.progress(plan) catch {
        error => raise tensor_parallel_transport_error(error)
      }
      match progress {
        ProcessCompletionReady => PhysicalExchangeCompleted
        ProcessAdvanced | ProcessPending => PhysicalExchangePending
      }
    }
  }
}

///|
fn WorkerPhysicalTransport::has_received(
  self : WorkerPhysicalTransport,
  sequence : UInt64,
) -> Bool raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) =>
      try process.received_for_sequence(sequence) catch {
        RootBoundFailure(Process(Invalid(Completion))) => false
        error => raise physical_transport_error(error)
      } noraise {
        _ => true
      }
    TensorParallelGroup(group) => group.has_received(sequence)
  }
}

///|
fn WorkerPhysicalTransport::received_frame(
  self : WorkerPhysicalTransport,
  sequence : UInt64,
) -> @worker_wire.ValidatedCompletionFrame raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) => {
      let received = process.received_for_sequence(sequence) catch {
        error => raise physical_transport_error(error)
      }
      process.validate_retirement(received) catch {
        error => raise physical_transport_error(error)
      }
      received.frame(process) catch {
        error => raise physical_transport_error(error)
      }
    }
    TensorParallelGroup(group) => {
      if !group.has_received(sequence) {
        raise Invalid(Sequence)
      }
      let received = group.received_completion() catch {
        error => raise tensor_parallel_transport_error(error)
      }
      group.completion_frame(received) catch {
        error => raise tensor_parallel_transport_error(error)
      }
    }
  }
}

///|
fn WorkerPhysicalTransport::supports_graph_telemetry(
  self : WorkerPhysicalTransport,
  sequence : UInt64,
) -> Bool {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(_) =>
      @worker_wire.worker_graph_telemetry_sample_required(sequence)
    TensorParallelGroup(_) => true
  }
}

///|
fn WorkerPhysicalTransport::received_graph_telemetry(
  self : WorkerPhysicalTransport,
  sequence : UInt64,
) -> @worker_wire.WorkerGraphTelemetry raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) => {
      let received = process.received_for_sequence(sequence) catch {
        error => raise physical_transport_error(error)
      }
      received.graph_telemetry(process) catch {
        error => raise physical_transport_error(error)
      }
    }
    TensorParallelGroup(group) => {
      if !group.has_received(sequence) {
        raise Invalid(Sequence)
      }
      let received = group.received_completion() catch {
        error => raise tensor_parallel_transport_error(error)
      }
      group.graph_telemetry(received) catch {
        error => raise tensor_parallel_transport_error(error)
      }
    }
  }
}

///|
fn WorkerPhysicalTransport::accept_completion(
  self : WorkerPhysicalTransport,
  sequence : UInt64,
  plan : @worker_protocol.SubmittedSchedulePlan,
  completion : @worker_protocol.SubmittedCompletion,
) -> Unit raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) => {
      let received = process.received_for_sequence(sequence) catch {
        error => raise physical_transport_error(error)
      }
      process.validate_retirement(received) catch {
        error => raise physical_transport_error(error)
      }
    }
    TensorParallelGroup(group) => {
      if !group.has_received(sequence) {
        raise Invalid(Sequence)
      }
      let received = group.received_completion() catch {
        error => raise tensor_parallel_transport_error(error)
      }
      ignore(
        group.accept_completion(received, plan, completion) catch {
          error => raise tensor_parallel_transport_error(error)
        },
      )
    }
  }
}

///|
fn WorkerPhysicalTransport::retire_accepted(
  self : WorkerPhysicalTransport,
  sequence : UInt64,
) -> Unit raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) => {
      let received = process.received_for_sequence(sequence) catch {
        error => raise physical_transport_error(error)
      }
      process.retire_received(received) catch {
        error => raise physical_transport_error(error)
      }
    }
    TensorParallelGroup(group) => {
      let accepted = group.accepted_completion_for_sequence(sequence) catch {
        error => raise tensor_parallel_transport_error(error)
      }
      group.retire_after_scheduler_commit(accepted) catch {
        error => raise tensor_parallel_transport_error(error)
      }
    }
  }
}

///|
fn WorkerPhysicalTransport::retire_received_failure(
  self : WorkerPhysicalTransport,
  sequence : UInt64,
) -> Unit raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) => {
      let received = process.received_for_sequence(sequence) catch {
        error => raise physical_transport_error(error)
      }
      process.validate_retirement(received) catch {
        error => raise physical_transport_error(error)
      }
      process.retire_received(received) catch {
        error => raise physical_transport_error(error)
      }
    }
    TensorParallelGroup(group) => {
      // A group completion can be retired only after exact service acceptance.
      // Recovery reaches this branch only for an already accepted completion.
      let accepted = group.accepted_completion_for_sequence(sequence) catch {
        error => raise tensor_parallel_transport_error(error)
      }
      group.retire_after_scheduler_commit(accepted) catch {
        error => raise tensor_parallel_transport_error(error)
      }
    }
  }
}

///|
fn WorkerPhysicalTransport::abandon_submission(
  self : WorkerPhysicalTransport,
  sequence : UInt64,
) -> Unit raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(process) => {
      let submission = process.submission_for_sequence(sequence) catch {
        error => raise physical_transport_error(error)
      }
      process.validate_abandonment(submission) catch {
        error => raise physical_transport_error(error)
      }
      process.abandon_submission(submission) catch {
        error => raise physical_transport_error(error)
      }
    }
    TensorParallelGroup(group) =>
      group.abandon_cleaned_submission(sequence) catch {
        error => raise tensor_parallel_transport_error(error)
      }
  }
}

///|
fn WorkerPhysicalTransport::retire_scheduler_only_failure(
  self : WorkerPhysicalTransport,
  sequence : UInt64,
) -> Unit raise WorkerServiceError {
  match self {
    Uninstalled =>
      abort("worker service physical transport installation invariant")
    RootedSingle(_) => ()
    TensorParallelGroup(group) =>
      group.retire_scheduler_only_failure(sequence) catch {
        error => raise tensor_parallel_transport_error(error)
      }
  }
}

///|
/// Advances a TP group's authenticated scheduler predecessor when a buffered
/// plan failed before physical submission. Transport-free owners exist only in
/// white-box scheduler tests; production-owned services must always retain a
/// physical transport.
fn WorkerService::retire_scheduler_only_failure(
  self : WorkerService,
  sequence : UInt64,
) -> Unit raise WorkerServiceError {
  match self.transport {
    Uninstalled =>
      if self.production_owned {
        abort("worker service physical transport installation invariant")
      }
    transport => transport.retire_scheduler_only_failure(sequence)
  }
}
