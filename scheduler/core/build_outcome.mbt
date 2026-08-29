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
    BuildPreempted => true
    _ => false
  }
}

///|
pub fn BuildNextOutcome::is_submitted(self : BuildNextOutcome) -> Bool {
  match self {
    BuildSubmitted(_, _) => true
    _ => false
  }
}
