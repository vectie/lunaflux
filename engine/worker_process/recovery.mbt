///|
/// Retires one exact unreturned submission after the old child has been fully
/// closed and the caller has committed its scheduler-side worker failure.
pub fn WorkerProcessSupervisor::abandon_submission(
  self : WorkerProcessSupervisor,
  submission : WorkerSubmission,
) -> Unit raise WorkerProcessError {
  self.validate_abandonment(submission)
  if submission.1 == 0 {
    self.state_a = 0
  } else {
    self.state_b = 0
  }
  self.last_retired = submission.2
}

///|
/// Nonmutating proof that one exact closed-child submission is the next
/// recoverable sequence.
pub fn WorkerProcessSupervisor::validate_abandonment(
  self : WorkerProcessSupervisor,
  submission : WorkerSubmission,
) -> Unit raise WorkerProcessError {
  if self.lifecycle != Closed {
    raise Invalid(Recovery)
  }
  submission.authenticate(self)
  if self.last_retired == 0xffffffffffffffffUL ||
    submission.2 != self.last_retired + 1UL {
    raise Invalid(Sequence)
  }
}

///|
/// Derives the only startup contract that can continue this sequence domain.
/// The old child must be closed and every exact submitted plan must already be
/// committed or failed by the scheduler and retired from this supervisor.
pub fn WorkerProcessSupervisor::recovery_startup_contract(
  self : WorkerProcessSupervisor,
) -> @worker_wire.WorkerStartupContract raise WorkerProcessError {
  if self.lifecycle != Closed ||
    self.outstanding_count() != 0 ||
    self.last_retired != self.last_submitted {
    raise Invalid(Recovery)
  }
  if self.last_submitted == 0xffffffffffffffffUL {
    raise Invalid(Sequence)
  }
  @worker_wire.WorkerStartupContract::new(
    identity=self.startup.identity(),
    bootstrap_digest=self.startup.bootstrap_digest(),
    model_generation_value=self.startup.model_generation_value(),
    worker_limits=self.startup.worker_limits(),
    inference_limits=self.startup.inference_limits(),
    has_predecessor=self.last_submitted != 0UL,
    predecessor_value=self.last_submitted,
  ) catch {
    error => raise Wire(error)
  }
}
