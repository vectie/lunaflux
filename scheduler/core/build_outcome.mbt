///|
pub fn BuildNextOutcome::is_idle(self : BuildNextOutcome) -> Bool {
  match self {
    BuildIdle => true
    _ => false
  }
}

///|
pub fn BuildNextOutcome::is_backpressured(self : BuildNextOutcome) -> Bool {
  match self {
    BuildBackpressured => true
    _ => false
  }
}

///|
pub fn BuildNextOutcome::is_preempted(self : BuildNextOutcome) -> Bool {
  match self {
    BuildPreempted(_, _) => true
    _ => false
  }
}

///|
/// Immutable identity of the discarded device history. Consumers with
/// request-keyed state must acknowledge its eviction before submitting a
/// replacement or replay. This is not a terminal request publication.
pub fn BuildNextOutcome::preempted_request_id(
  self : Self,
) -> UInt64 raise SchedulerError {
  match self {
    BuildPreempted(id, _) => id
    _ => raise invalid(Plan, NoRunnableWork)
  }
}

///|
pub fn BuildNextOutcome::preempted_request_generation(
  self : Self,
) -> UInt64 raise SchedulerError {
  match self {
    BuildPreempted(_, generation) => generation
    _ => raise invalid(Plan, NoRunnableWork)
  }
}

///|
pub fn BuildNextOutcome::is_submitted(self : BuildNextOutcome) -> Bool {
  match self {
    BuildSubmitted(_, _) => true
    _ => false
  }
}
