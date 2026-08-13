///|
fn wire_failure(
  value : Int,
) -> @worker_protocol.WorkerFailure raise SchedulerError {
  match value {
    0 => ExecutionFailed
    1 => ResourceUnavailable
    2 => CancelledBeforeExecution
    _ => raise WorkerWire(InvalidFrame(CompletionEntry))
  }
}

///|
fn append_wire_prefill(
  writer : @worker_protocol.CompletionWriter,
  plan : @worker_protocol.SubmittedSchedulePlan,
  frame : @worker_wire.ValidatedCompletionFrame,
  index : Int,
) -> Unit raise SchedulerError {
  let row = plan.prefill_row(index) catch {
    error => raise WorkerProtocol(error)
  }
  let slot = row.completion_slot() catch {
    error => raise WorkerProtocol(error)
  }
  let entry = frame.entry_for_slot(slot.value()) catch {
    error => raise WorkerWire(error)
  }
  let request_id = row.request_id() catch {
    error => raise WorkerProtocol(error)
  }
  let generation = row.request_generation() catch {
    error => raise WorkerProtocol(error)
  }
  let kind = entry.kind_value() catch { error => raise WorkerWire(error) }
  match kind {
    0 => {
      let processed = entry.processed_tokens() catch {
        error => raise WorkerWire(error)
      }
      writer.append_prefill_completed(
        slot~,
        request_id~,
        request_generation=generation,
        processed_tokens=processed,
      ) catch {
        error => raise WorkerProtocol(error)
      }
    }
    1 => {
      let processed = entry.processed_tokens() catch {
        error => raise WorkerWire(error)
      }
      let token = entry.token_id() catch { error => raise WorkerWire(error) }
      writer.append_final_prefill_completed(
        slot~,
        request_id~,
        request_generation=generation,
        processed_tokens=processed,
        token_id=token,
      ) catch {
        error => raise WorkerProtocol(error)
      }
    }
    3 => {
      let failure_value = entry.failure_value() catch {
        error => raise WorkerWire(error)
      }
      let failure = wire_failure(failure_value)
      writer.append_failed(
        slot~,
        request_id~,
        request_generation=generation,
        failure~,
      ) catch {
        error => raise WorkerProtocol(error)
      }
    }
    _ => raise WorkerWire(InvalidFrame(CompletionEntry))
  }
}

///|
fn append_wire_decode(
  writer : @worker_protocol.CompletionWriter,
  plan : @worker_protocol.SubmittedSchedulePlan,
  frame : @worker_wire.ValidatedCompletionFrame,
  index : Int,
) -> Unit raise SchedulerError {
  let row = plan.decode_row(index) catch {
    error => raise WorkerProtocol(error)
  }
  let slot = row.completion_slot() catch {
    error => raise WorkerProtocol(error)
  }
  let entry = frame.entry_for_slot(slot.value()) catch {
    error => raise WorkerWire(error)
  }
  let request_id = row.request_id() catch {
    error => raise WorkerProtocol(error)
  }
  let generation = row.request_generation() catch {
    error => raise WorkerProtocol(error)
  }
  let kind = entry.kind_value() catch { error => raise WorkerWire(error) }
  match kind {
    2 => {
      let token = entry.token_id() catch { error => raise WorkerWire(error) }
      writer.append_decode_completed(
        slot~,
        request_id~,
        request_generation=generation,
        token_id=token,
      ) catch {
        error => raise WorkerProtocol(error)
      }
    }
    3 => {
      let failure_value = entry.failure_value() catch {
        error => raise WorkerWire(error)
      }
      let failure = wire_failure(failure_value)
      writer.append_failed(
        slot~,
        request_id~,
        request_generation=generation,
        failure~,
      ) catch {
        error => raise WorkerProtocol(error)
      }
    }
    _ => raise WorkerWire(InvalidFrame(CompletionEntry))
  }
}

///|
fn validate_submitted_matches_frame(
  completion : @worker_protocol.SubmittedCompletion,
  plan : @worker_protocol.SubmittedSchedulePlan,
  frame : @worker_wire.ValidatedCompletionFrame,
) -> Unit raise SchedulerError {
  let prefill_count = plan.prefill_count() catch {
    error => raise WorkerProtocol(error)
  }
  let decode_count = plan.decode_count() catch {
    error => raise WorkerProtocol(error)
  }
  for row_index in 0..<prefill_count {
    let row = plan.prefill_row(row_index) catch {
      error => raise WorkerProtocol(error)
    }
    let slot = row.completion_slot() catch {
      error => raise WorkerProtocol(error)
    }
    let actual = completion.entry_for_slot(slot) catch {
      error => raise WorkerProtocol(error)
    }
    let expected = frame.entry_for_slot(slot.value()) catch {
      error => raise WorkerWire(error)
    }
    let expected_kind = expected.kind_value() catch {
      error => raise WorkerWire(error)
    }
    let actual_kind = actual.kind() catch {
      error => raise WorkerProtocol(error)
    }
    match (expected_kind, actual_kind) {
      (0, PrefillCompletion) => {
        let actual_processed = actual.processed_tokens() catch {
          error => raise WorkerProtocol(error)
        }
        let expected_processed = expected.processed_tokens() catch {
          error => raise WorkerWire(error)
        }
        if actual_processed != expected_processed {
          raise WorkerWire(InvalidFrame(CompletionEntry))
        }
      }
      (1, FinalPrefillCompletion) => {
        let actual_processed = actual.processed_tokens() catch {
          error => raise WorkerProtocol(error)
        }
        let expected_processed = expected.processed_tokens() catch {
          error => raise WorkerWire(error)
        }
        let actual_token = actual.token_id() catch {
          error => raise WorkerProtocol(error)
        }
        let expected_token = expected.token_id() catch {
          error => raise WorkerWire(error)
        }
        if actual_processed != expected_processed ||
          actual_token != expected_token {
          raise WorkerWire(InvalidFrame(CompletionEntry))
        }
      }
      (3, FailedCompletion) => {
        let actual_failure = actual.failure() catch {
          error => raise WorkerProtocol(error)
        }
        let expected_failure_value = expected.failure_value() catch {
          error => raise WorkerWire(error)
        }
        if actual_failure != wire_failure(expected_failure_value) {
          raise WorkerWire(InvalidFrame(CompletionEntry))
        }
      }
      _ => raise WorkerWire(InvalidFrame(CompletionEntry))
    }
  }
  for decode_index in 0..<decode_count {
    let row = plan.decode_row(decode_index) catch {
      error => raise WorkerProtocol(error)
    }
    let slot = row.completion_slot() catch {
      error => raise WorkerProtocol(error)
    }
    let actual = completion.entry_for_slot(slot) catch {
      error => raise WorkerProtocol(error)
    }
    let expected = frame.entry_for_slot(slot.value()) catch {
      error => raise WorkerWire(error)
    }
    let expected_kind = expected.kind_value() catch {
      error => raise WorkerWire(error)
    }
    let actual_kind = actual.kind() catch {
      error => raise WorkerProtocol(error)
    }
    match (expected_kind, actual_kind) {
      (2, DecodeCompletion) => {
        let actual_token = actual.token_id() catch {
          error => raise WorkerProtocol(error)
        }
        let expected_token = expected.token_id() catch {
          error => raise WorkerWire(error)
        }
        if actual_token != expected_token {
          raise WorkerWire(InvalidFrame(CompletionEntry))
        }
      }
      (3, FailedCompletion) => {
        let actual_failure = actual.failure() catch {
          error => raise WorkerProtocol(error)
        }
        let expected_failure_value = expected.failure_value() catch {
          error => raise WorkerWire(error)
        }
        if actual_failure != wire_failure(expected_failure_value) {
          raise WorkerWire(InvalidFrame(CompletionEntry))
        }
      }
      _ => raise WorkerWire(InvalidFrame(CompletionEntry))
    }
  }
}

///|
/// Converts one fully validated worker-process frame into the scheduler-owned
/// completion buffer paired with `plan`. This does not retire the plan or
/// mutate request/KV state. The caller retains both returned capabilities and
/// retries `complete` if publication backpressure rejects retirement.
pub fn Scheduler::accept_completion_frame(
  self : Scheduler,
  plan : @worker_protocol.SubmittedSchedulePlan,
  frame : @worker_wire.ValidatedCompletionFrame,
) -> @worker_protocol.SubmittedCompletion raise SchedulerError {
  frame.validate_for(plan) catch {
    error => raise WorkerWire(error)
  }
  let side = self.completion_side(plan)
  let owner = if side == 0 { self.completion_a } else { self.completion_b }
  if owner.state() == CompletionSubmitted {
    let completion = owner.current_submitted(plan) catch {
      error => raise WorkerProtocol(error)
    }
    validate_submitted_matches_frame(completion, plan, frame)
    return completion
  }
  let writer = self.completion_writer(plan)
  try {
    let prefill_count = plan.prefill_count() catch {
      error => raise WorkerProtocol(error)
    }
    let decode_count = plan.decode_count() catch {
      error => raise WorkerProtocol(error)
    }
    for index in 0..<prefill_count {
      append_wire_prefill(writer, plan, frame, index)
    }
    for index in 0..<decode_count {
      append_wire_decode(writer, plan, frame, index)
    }
    writer.submit() catch {
      error => raise WorkerProtocol(error)
    }
  } catch {
    error => {
      writer.abort() catch {
        cleanup => raise WorkerProtocol(cleanup)
      }
      raise error
    }
  }
}
