///|
fn append_bridge_prefill(
  writer : @worker_protocol.CompletionWriter,
  plan : @worker_protocol.SubmittedSchedulePlan,
  frame : @worker_wire.ValidatedCompletionFrame,
  index : Int,
) -> Unit raise WorkerServiceError {
  let row = plan.prefill_row(index) catch { _ => raise Invalid(WorkerOutput) }
  let slot = row.completion_slot() catch { _ => raise Invalid(WorkerOutput) }
  let request_id = row.request_id() catch { _ => raise Invalid(WorkerOutput) }
  let request_generation = row.request_generation() catch {
    _ => raise Invalid(WorkerOutput)
  }
  let entry = frame.entry_for_slot(slot.value()) catch {
    _ => raise Invalid(WorkerOutput)
  }
  let kind = entry.kind() catch { _ => raise Invalid(WorkerOutput) }
  match kind {
    PrefillCompletion => {
      let processed_tokens = entry.processed_tokens() catch {
        _ => raise Invalid(WorkerOutput)
      }
      writer.append_prefill_completed(
        slot~,
        request_id~,
        request_generation~,
        processed_tokens~,
      ) catch {
        _ => raise Invalid(WorkerOutput)
      }
    }
    FinalPrefillCompletion => {
      let processed_tokens = entry.processed_tokens() catch {
        _ => raise Invalid(WorkerOutput)
      }
      let token_id = entry.token_id() catch { _ => raise Invalid(WorkerOutput) }
      writer.append_final_prefill_completed(
        slot~,
        request_id~,
        request_generation~,
        processed_tokens~,
        token_id~,
      ) catch {
        _ => raise Invalid(WorkerOutput)
      }
    }
    FailedCompletion => {
      let failure = entry.failure() catch { _ => raise Invalid(WorkerOutput) }
      writer.append_failed(slot~, request_id~, request_generation~, failure~) catch {
        _ => raise Invalid(WorkerOutput)
      }
    }
    DecodeCompletion => raise Invalid(WorkerOutput)
  }
}

///|
fn append_bridge_decode(
  writer : @worker_protocol.CompletionWriter,
  plan : @worker_protocol.SubmittedSchedulePlan,
  frame : @worker_wire.ValidatedCompletionFrame,
  index : Int,
) -> Unit raise WorkerServiceError {
  let row = plan.decode_row(index) catch { _ => raise Invalid(WorkerOutput) }
  let slot = row.completion_slot() catch { _ => raise Invalid(WorkerOutput) }
  let request_id = row.request_id() catch { _ => raise Invalid(WorkerOutput) }
  let request_generation = row.request_generation() catch {
    _ => raise Invalid(WorkerOutput)
  }
  let entry = frame.entry_for_slot(slot.value()) catch {
    _ => raise Invalid(WorkerOutput)
  }
  let kind = entry.kind() catch { _ => raise Invalid(WorkerOutput) }
  match kind {
    DecodeCompletion => {
      let token_id = entry.token_id() catch { _ => raise Invalid(WorkerOutput) }
      writer.append_decode_completed(
        slot~,
        request_id~,
        request_generation~,
        token_id~,
      ) catch {
        _ => raise Invalid(WorkerOutput)
      }
    }
    FailedCompletion => {
      let failure = entry.failure() catch { _ => raise Invalid(WorkerOutput) }
      writer.append_failed(slot~, request_id~, request_generation~, failure~) catch {
        _ => raise Invalid(WorkerOutput)
      }
    }
    PrefillCompletion | FinalPrefillCompletion => raise Invalid(WorkerOutput)
  }
}

///|
/// Copies a frame that was already authenticated before writer acquisition.
/// Any later rejection aborts the exact writer epoch before returning a bounded
/// worker-output category.
fn submit_bridge_after_open(
  writer : @worker_protocol.CompletionWriter,
  plan : @worker_protocol.SubmittedSchedulePlan,
  frame : @worker_wire.ValidatedCompletionFrame,
) -> @worker_protocol.SubmittedCompletion raise WorkerServiceError {
  try {
    let prefill_count = plan.prefill_count() catch {
      _ => raise Invalid(WorkerOutput)
    }
    let decode_count = plan.decode_count() catch {
      _ => raise Invalid(WorkerOutput)
    }
    for index in 0..<prefill_count {
      append_bridge_prefill(writer, plan, frame, index)
    }
    for index in 0..<decode_count {
      append_bridge_decode(writer, plan, frame, index)
    }
    writer.submit() catch {
      _ => raise Invalid(WorkerOutput)
    }
  } catch {
    _ => {
      writer.abort() catch {
        _ => raise Invalid(WorkerOutput)
      }
      raise Invalid(WorkerOutput)
    }
  }
}

///|
/// Authenticates untrusted wire evidence before acquiring the scheduler's
/// exclusive completion writer, then transactionally copies only typed values.
fn bridge_completion(
  scheduler : @core.Scheduler,
  plan : @worker_protocol.SubmittedSchedulePlan,
  frame : @worker_wire.ValidatedCompletionFrame,
) -> @worker_protocol.SubmittedCompletion raise WorkerServiceError {
  frame.validate_for(plan) catch {
    _ => raise Invalid(WorkerOutput)
  }
  let writer = scheduler.completion_writer(plan) catch {
    WorkerProtocol(_) => raise Invalid(WorkerOutput)
    error => raise scheduler_error(error)
  }
  submit_bridge_after_open(writer, plan, frame)
}
