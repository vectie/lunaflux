///|
/// Resolves a submitted outcome back to the exact authenticated worker handle
/// without exposing either mutable A/B plan owner.
pub fn Scheduler::submitted_plan(
  self : Scheduler,
  outcome : BuildNextOutcome,
) -> @worker_protocol.SubmittedSchedulePlan raise SchedulerError {
  match outcome {
    BuildIdle => raise invalid(Plan, NoRunnableWork)
    BuildBackpressured => raise invalid(Plan, Backpressured)
    BuildSubmitted(side, sequence) => {
      let owner = if side == 0 {
        if self.plan_a_sequence != sequence {
          raise invalid(Plan, Stale)
        }
        self.plan_a
      } else if side == 1 {
        if self.plan_b_sequence != sequence {
          raise invalid(Plan, Stale)
        }
        self.plan_b
      } else {
        raise invalid(Plan, Mismatch)
      }
      try owner.current_submitted() catch {
        error => raise WorkerProtocol(error)
      } noraise {
        plan => plan
      }
    }
  }
}

///|
/// Resolves one exact still-submitted sequence without exposing A/B ownership.
pub fn Scheduler::submitted_plan_for_sequence(
  self : Scheduler,
  sequence : UInt64,
) -> @worker_protocol.SubmittedSchedulePlan raise SchedulerError {
  let owner = if sequence != 0UL && self.plan_a_sequence == sequence {
    self.plan_a
  } else if sequence != 0UL && self.plan_b_sequence == sequence {
    self.plan_b
  } else {
    raise invalid(Plan, Stale)
  }
  owner.current_submitted() catch {
    error => raise WorkerProtocol(error)
  }
}
