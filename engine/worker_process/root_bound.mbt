///|
fn root_failure(
  error : @approved_fs.ApprovedFsError,
) -> RootBoundWorkerProcessFailure {
  match error {
    Busy(_) => RootsBusy
    _ => RootsCloseFailed
  }
}

///|
fn close_root_pair(
  roots : @approved_fs.WorkerApprovedRoots,
) -> RootBoundWorkerProcessFailure? {
  try @approved_fs.close_worker_approved_roots(roots) catch {
    error => Some(root_failure(error))
  } noraise {
    _ => None
  }
}

///|
fn roots_still_live(roots : @approved_fs.WorkerApprovedRoots) -> Bool {
  !@approved_fs.worker_approved_roots_is_closed(roots)
}

///|
/// Acquires a private duplicate pair from caller-owned roots and uses it for
/// the initial child and every replacement. The input roots are only borrowed;
/// no public API accepts or returns the retained pair.
pub fn prepare_with_approved_roots(
  executable : Bytes,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  limits : WorkerProcessLimits,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
) -> RootBoundWorkerProcessPreparation raise RootBoundWorkerProcessError {
  if source.digest() != startup.bootstrap_source_digest() {
    raise RootBoundFailure(Process(Invalid(SourceMismatch)))
  }
  let roots = @approved_fs.acquire_worker_approved_roots(
    model_root~,
    kernel_root~,
  ) catch {
    error => raise RootBoundFailure(root_failure(error))
  }
  let startup_result = prepare_internal(
    executable,
    startup,
    source,
    limits,
    Some(roots),
  ) catch {
    error => {
      let roots_failure = close_root_pair(roots)
      if roots_still_live(roots) {
        return RootBoundCleanupRequired({
          roots,
          child: None,
          cause: Process(error),
          child_cleanup_failure: None,
          roots_cleanup_failure: roots_failure,
          closed: false,
        })
      }
      if roots_failure is Some(cleanup) {
        raise RootBoundCompoundFailure(Process(error), cleanup)
      }
      raise RootBoundFailure(Process(error))
    }
  }
  match startup_result {
    ReadyProcess(process) =>
      ReadyRootBoundProcess({
        executable,
        limits,
        roots,
        process,
        failed_replacement: None,
        cause: None,
        child_cleanup_failure: None,
        roots_cleanup_failure: None,
        lifecycle: RootBoundReady,
      })
    CleanupRequired(child) => {
      let roots_failure = close_root_pair(roots)
      RootBoundCleanupRequired({
        roots,
        child: Some(child),
        cause: Process(child.cause()),
        child_cleanup_failure: Some(Process(child.cleanup_failure())),
        roots_cleanup_failure: roots_failure,
        closed: false,
      })
    }
  }
}

///|
/// Exhaustively retries both independent initial-startup authorities. A native
/// root close failure that consumed the pair remains evidence, not false retry
/// authority; Busy retains this owner for another attempt.
pub fn FailedRootBoundWorkerProcessStartup::retry_close(
  self : FailedRootBoundWorkerProcessStartup,
) -> Unit raise RootBoundWorkerProcessError {
  if self.closed {
    return
  }
  let attempt = attempt_root_bound_cleanup(
    self.child is Some(_),
    roots_still_live(self.roots),
    () => {
      guard self.child is Some(child) else { return None }
      try child.retry_close() catch {
        error => Some(Process(error))
      } noraise {
        _ => {
          self.child = None
          None
        }
      }
    },
    () => close_root_pair(self.roots),
    () => roots_still_live(self.roots),
  )
  self.child_cleanup_failure = attempt.child_failure
  self.roots_cleanup_failure = attempt.roots_failure
  self.closed = !attempt.child_live && !attempt.roots_live
  raise_cleanup_attempt(attempt)
}

///|
pub fn RootBoundWorkerProcessSupervisor::bootstrap_source(
  self : RootBoundWorkerProcessSupervisor,
) -> @worker_wire.EncodedBootstrapSource {
  self.process.bootstrap_source()
}

///|
pub fn RootBoundWorkerProcessSupervisor::startup_contract(
  self : RootBoundWorkerProcessSupervisor,
) -> @worker_wire.WorkerStartupContract {
  self.process.startup_contract()
}

///|
pub fn RootBoundWorkerProcessSupervisor::has_capacity(
  self : RootBoundWorkerProcessSupervisor,
) -> Bool {
  self.lifecycle == RootBoundReady &&
  self.process.outstanding_count() == 0 &&
  self.process.has_capacity()
}

///|
pub fn RootBoundWorkerProcessSupervisor::outstanding_count(
  self : RootBoundWorkerProcessSupervisor,
) -> Int {
  self.process.outstanding_count()
}

///|
pub fn RootBoundWorkerProcessSupervisor::shutdown_clean(
  self : RootBoundWorkerProcessSupervisor,
) -> Unit raise RootBoundWorkerProcessError {
  if self.lifecycle != RootBoundReady {
    raise RootBoundInvalidLifecycle
  }
  if self.process.outstanding_count() != 0 {
    raise RootBoundFailure(Process(Backpressured))
  }
  self.process.shutdown_clean() catch {
    error => {
      self.cause = Some(Process(error))
      self.lifecycle = RootBoundRecoveryRequired
      raise RootBoundFailure(Process(error))
    }
  }
}

///|
pub fn RootBoundWorkerProcessSupervisor::recovery_startup_contract(
  self : RootBoundWorkerProcessSupervisor,
) -> @worker_wire.WorkerStartupContract raise RootBoundWorkerProcessError {
  if self.lifecycle != RootBoundRecovering {
    raise RootBoundInvalidLifecycle
  }
  self.process.recovery_startup_contract() catch {
    error => raise RootBoundFailure(Process(error))
  }
}

///|
pub fn RootBoundWorkerProcessSupervisor::submit(
  self : RootBoundWorkerProcessSupervisor,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> RootBoundWorkerSubmission raise RootBoundWorkerProcessError {
  if self.lifecycle != RootBoundReady {
    raise RootBoundInvalidLifecycle
  }
  if self.process.outstanding_count() != 0 {
    raise RootBoundFailure(Process(Backpressured))
  }
  let submitted = self.process.submit(plan) catch {
    error => {
      if self.process.lifecycle() != Ready {
        self.cause = Some(Process(error))
        self.lifecycle = RootBoundRecoveryRequired
      }
      raise RootBoundFailure(Process(error))
    }
  }
  RootBoundWorkerSubmission(self, submitted.1, submitted.2)
}

///|
pub fn RootBoundWorkerSubmission::sequence_value(
  self : RootBoundWorkerSubmission,
) -> UInt64 {
  self.2
}

///|
fn RootBoundWorkerSubmission::authenticate(
  self : RootBoundWorkerSubmission,
  owner : RootBoundWorkerProcessSupervisor,
) -> Unit raise RootBoundWorkerProcessError {
  if !physical_equal(self.0, owner) {
    raise RootBoundOwnerMismatch
  }
}

///|
pub fn RootBoundWorkerProcessSupervisor::receive(
  self : RootBoundWorkerProcessSupervisor,
  submission : RootBoundWorkerSubmission,
  plan : @worker_protocol.SubmittedSchedulePlan,
) -> RootBoundReceivedWorkerCompletion raise RootBoundWorkerProcessError {
  submission.authenticate(self)
  let received = self.process.receive(
    WorkerSubmission(self.process, submission.1, submission.2),
    plan,
  ) catch {
    error => {
      if self.process.lifecycle() != Ready {
        self.cause = Some(Process(error))
        self.lifecycle = RootBoundRecoveryRequired
      }
      raise RootBoundFailure(Process(error))
    }
  }
  RootBoundReceivedWorkerCompletion(self, received.1, received.2)
}

///|
pub fn RootBoundWorkerProcessSupervisor::submission_for_sequence(
  self : RootBoundWorkerProcessSupervisor,
  sequence : UInt64,
) -> RootBoundWorkerSubmission raise RootBoundWorkerProcessError {
  let submission = self.process.submission_for_sequence(sequence) catch {
    error => raise RootBoundFailure(Process(error))
  }
  RootBoundWorkerSubmission(self, submission.1, submission.2)
}

///|
pub fn RootBoundWorkerProcessSupervisor::received_for_sequence(
  self : RootBoundWorkerProcessSupervisor,
  sequence : UInt64,
) -> RootBoundReceivedWorkerCompletion raise RootBoundWorkerProcessError {
  let received = self.process.received_for_sequence(sequence) catch {
    error => raise RootBoundFailure(Process(error))
  }
  RootBoundReceivedWorkerCompletion(self, received.1, received.2)
}

///|
pub fn RootBoundReceivedWorkerCompletion::sequence_value(
  self : RootBoundReceivedWorkerCompletion,
) -> UInt64 {
  self.2
}

///|
pub fn RootBoundReceivedWorkerCompletion::frame(
  self : RootBoundReceivedWorkerCompletion,
  owner : RootBoundWorkerProcessSupervisor,
) -> @worker_wire.ValidatedCompletionFrame raise RootBoundWorkerProcessError {
  if !physical_equal(self.0, owner) {
    raise RootBoundOwnerMismatch
  }
  ReceivedWorkerCompletion(owner.process, self.1, self.2).frame(owner.process) catch {
    error => raise RootBoundFailure(Process(error))
  }
}

///|
pub fn RootBoundWorkerProcessSupervisor::validate_retirement(
  self : RootBoundWorkerProcessSupervisor,
  received : RootBoundReceivedWorkerCompletion,
) -> Unit raise RootBoundWorkerProcessError {
  if !physical_equal(received.0, self) {
    raise RootBoundOwnerMismatch
  }
  self.process.validate_retirement(
    ReceivedWorkerCompletion(self.process, received.1, received.2),
  ) catch {
    error => raise RootBoundFailure(Process(error))
  }
}

///|
pub fn RootBoundWorkerProcessSupervisor::retire_received(
  self : RootBoundWorkerProcessSupervisor,
  received : RootBoundReceivedWorkerCompletion,
) -> Unit raise RootBoundWorkerProcessError {
  if !physical_equal(received.0, self) {
    raise RootBoundOwnerMismatch
  }
  self.process.retire_received(
    ReceivedWorkerCompletion(self.process, received.1, received.2),
  ) catch {
    error => raise RootBoundFailure(Process(error))
  }
}

///|
pub fn RootBoundWorkerProcessSupervisor::validate_abandonment(
  self : RootBoundWorkerProcessSupervisor,
  submission : RootBoundWorkerSubmission,
) -> Unit raise RootBoundWorkerProcessError {
  submission.authenticate(self)
  self.process.validate_abandonment(
    WorkerSubmission(self.process, submission.1, submission.2),
  ) catch {
    error => raise RootBoundFailure(Process(error))
  }
}

///|
pub fn RootBoundWorkerProcessSupervisor::abandon_submission(
  self : RootBoundWorkerProcessSupervisor,
  submission : RootBoundWorkerSubmission,
) -> Unit raise RootBoundWorkerProcessError {
  submission.authenticate(self)
  self.process.abandon_submission(
    WorkerSubmission(self.process, submission.1, submission.2),
  ) catch {
    error => raise RootBoundFailure(Process(error))
  }
}
