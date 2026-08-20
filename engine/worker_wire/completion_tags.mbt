///|
fn completion_kind_to_wire(kind : @worker_protocol.CompletionEntryKind) -> UInt {
  match kind {
    PrefillCompletion => 0
    FinalPrefillCompletion => 1
    DecodeCompletion => 2
    FailedCompletion => 3
  }
}

///|
fn completion_kind_from_wire(
  value : UInt,
) -> @worker_protocol.CompletionEntryKind raise WorkerWireError {
  match value {
    0 => PrefillCompletion
    1 => FinalPrefillCompletion
    2 => DecodeCompletion
    3 => FailedCompletion
    _ => raise InvalidFrame(CompletionEntry)
  }
}

///|
fn worker_failure_to_wire(failure : @worker_protocol.WorkerFailure) -> UInt {
  match failure {
    ExecutionFailed => 0
    ResourceUnavailable => 1
    CancelledBeforeExecution => 2
  }
}

///|
fn worker_failure_from_wire(
  value : UInt,
) -> @worker_protocol.WorkerFailure raise WorkerWireError {
  match value {
    0 => ExecutionFailed
    1 => ResourceUnavailable
    2 => CancelledBeforeExecution
    _ => raise InvalidFrame(CompletionEntry)
  }
}
