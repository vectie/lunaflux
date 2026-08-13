///|
fn append_worker_failure_rows(
  writer : @worker_protocol.CompletionWriter,
  plan : @worker_protocol.SubmittedSchedulePlan,
  failure : @worker_protocol.WorkerFailure,
) -> Unit raise SchedulerError {
  let prefill_count = plan.prefill_count() catch {
    error => raise WorkerProtocol(error)
  }
  for index in 0..<prefill_count {
    let row = plan.prefill_row(index) catch {
      error => raise WorkerProtocol(error)
    }
    writer.append_failed(
      slot=row.completion_slot(),
      request_id=row.request_id(),
      request_generation=row.request_generation(),
      failure~,
    ) catch {
      error => raise WorkerProtocol(error)
    }
  }
  let decode_count = plan.decode_count() catch {
    error => raise WorkerProtocol(error)
  }
  for index in 0..<decode_count {
    let row = plan.decode_row(index) catch {
      error => raise WorkerProtocol(error)
    }
    writer.append_failed(
      slot=row.completion_slot(),
      request_id=row.request_id(),
      request_generation=row.request_generation(),
      failure~,
    ) catch {
      error => raise WorkerProtocol(error)
    }
  }
}

///|
/// Commits one exact submitted plan as a bounded worker failure. If scheduler
/// publication is backpressured, the completed failure epoch remains submitted
/// and the same call reuses it on retry without reopening a writer.
pub fn Scheduler::fail_plan(
  self : Scheduler,
  plan : @worker_protocol.SubmittedSchedulePlan,
  failure : @worker_protocol.WorkerFailure,
) -> Unit raise SchedulerError {
  let side = self.completion_side(plan)
  let owner = if side == 0 { self.completion_a } else { self.completion_b }
  let completion = match owner.state() {
    CompletionWritable => {
      let writer = self.completion_writer(plan)
      append_worker_failure_rows(writer, plan, failure) catch {
        error => {
          writer.abort() catch {
            abort_error => raise WorkerProtocol(abort_error)
          }
          raise error
        }
      }
      writer.submit() catch {
        error => raise WorkerProtocol(error)
      }
    }
    CompletionSubmitted =>
      owner.current_submitted(plan) catch {
        error => raise WorkerProtocol(error)
      }
    Consumed => raise invalid(Completion, Stale)
  }
  self.complete(plan, completion)
}
