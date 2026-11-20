///|
fn WorkerPhysicalTransport::begin_exchange(
  self : WorkerPhysicalTransport,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> UInt64 raise WorkerServiceError {
  match self {
    RemoteGroup(group) => {
      group.current.begin(plan) catch {
        error => raise remote_error(error)
      }
      plan.sequence_wire() catch {
        error => raise Scheduler(WorkerProtocol(error))
      }
    }
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
    RemoteGroup(group) => group.current.has_submission(sequence)
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
    RemoteGroup(group) => {
      let done = group.current.progress(plan) catch {
        error => raise remote_error(error)
      }
      if done {
        PhysicalExchangeCompleted
      } else {
        PhysicalExchangePending
      }
    }
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
    RemoteGroup(group) => group.current.has_received(sequence)
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
    RemoteGroup(group) =>
      group.current.completion_frame(sequence) catch {
        error => raise remote_error(error)
      }
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
    RemoteGroup(group) => {
      ignore(group)
      true
    }
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
    RemoteGroup(group) =>
      group.current.graph_telemetry(sequence) catch {
        error => raise remote_error(error)
      }
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
    RemoteGroup(group) => {
      if !group.current.has_received(sequence) {
        raise Invalid(Sequence)
      }
      group.current.accept_completion(plan, completion) catch {
        error => raise remote_error(error)
      }
    }
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
    RemoteGroup(group) => {
      group.current.retire_after_scheduler_commit(sequence) catch {
        error => raise remote_error(error)
      }
      group.previous = sequence
    }
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
    RemoteGroup(group) => {
      group.current.retire_after_scheduler_commit(sequence) catch {
        error => raise remote_error(error)
      }
      group.previous = sequence
    }
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
    RemoteGroup(group) => group.retire_failed(sequence)
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
    RemoteGroup(group) => group.retire_failed(sequence)
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
