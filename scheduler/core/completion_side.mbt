///|
fn Scheduler::completion_side(
  self : Scheduler,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> Int raise SchedulerError {
  let sequence = try plan.sequence_wire() catch {
    error => raise WorkerProtocol(error)
  } noraise {
    value => value
  }
  if sequence == self.plan_a_sequence {
    self.plan_a.validate_submitted(plan) catch {
      error => raise WorkerProtocol(error)
    }
    0
  } else if sequence == self.plan_b_sequence {
    self.plan_b.validate_submitted(plan) catch {
      error => raise WorkerProtocol(error)
    }
    1
  } else {
    raise invalid(Completion, Stale)
  }
}

///|
/// Issues a worker-only writer for the completion owner paired with the exact
/// scheduler-owned plan side and epoch. The mutable owner stays private.
pub fn Scheduler::completion_writer(
  self : Scheduler,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> @worker_protocol.CompletionWriter raise SchedulerError {
  self.require_no_cancel_reservation()
  let side = self.completion_side(plan)
  try {
    if side == 0 {
      self.completion_a.writer_for(plan)
    } else {
      self.completion_b.writer_for(plan)
    }
  } catch {
    error => raise WorkerProtocol(error)
  }
}
