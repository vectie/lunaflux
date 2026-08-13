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
fn close_root_pair_status(roots : @approved_fs.WorkerApprovedRoots) -> Int {
  @approved_fs.worker_approved_roots_close_status(roots)
}

///|
fn root_failure_status(status : Int) -> RootBoundWorkerProcessFailure? {
  if status == 0 {
    None
  } else if status == 7 {
    Some(RootsBusy)
  } else {
    Some(RootsCloseFailed)
  }
}

///|
fn roots_still_live(roots : @approved_fs.WorkerApprovedRoots) -> Bool {
  !@approved_fs.worker_approved_roots_is_closed(roots)
}

///|
fn RootBoundWorkerProcessSupervisor::require_roots(
  self : RootBoundWorkerProcessSupervisor,
) -> @approved_fs.WorkerApprovedRoots {
  if !self.roots_selected {
    abort("root-bound roots selection invariant")
  }
  self.roots
}

///|
/// Acquires a private duplicate pair from caller-owned roots and uses it for
/// the initial child and every replacement. The input roots are only borrowed;
/// no public API accepts or returns the retained pair.
fn prepare_root_bound(
  executable : Bytes,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  limits : WorkerProcessLimits,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  exchange_claimed : Bool,
  expected_predecessor : UInt64,
) -> RootBoundWorkerProcessPreparation raise RootBoundWorkerProcessError {
  if source.digest() != startup.bootstrap_source_digest() {
    raise RootBoundFailure(Process(Invalid(SourceMismatch)))
  }
  if exchange_claimed &&
    (
      startup.has_predecessor() != (expected_predecessor != 0UL) ||
      startup.predecessor_value() != expected_predecessor
    ) {
    raise RootBoundFailure(Process(Invalid(Sequence)))
  }
  let prepared = preflight_prepare(executable, startup, source, limits) catch {
    error => raise RootBoundFailure(Process(error))
  }
  let prepared_roots = @approved_fs.prepare_worker_approved_roots()
  let roots = prepared_roots.root_owner()
  let ready_owner : RootBoundWorkerProcessSupervisor = {
    executable: prepared.executable,
    limits,
    roots,
    roots_selected: false,
    process: prepared.supervisor,
    failed_replacement: None,
    cause: None,
    child_cleanup_failure: None,
    roots_cleanup_failure: None,
    transport_mode: if exchange_claimed {
      2
    } else {
      0
    },
    exchange_epoch: 0UL,
    active_exchange_epoch: 0UL,
    exchange_phase: -1,
    exchange_side: -1,
    exchange_sequence: 0UL,
    lifecycle: RootBoundReady,
  }
  let failed_owner : FailedRootBoundWorkerProcessStartup = {
    roots,
    roots_selected: false,
    child: None,
    child_cleanup_process: None,
    child_cleanup_native: None,
    roots_cleanup_status: 0,
    closed: false,
  }
  let root_activation_status = @approved_fs.acquire_prepared_worker_approved_roots_status(
    prepared_roots, model_root, kernel_root,
  )
  if root_activation_status != 0 {
    raise RootBoundFailure(
      if root_activation_status == 7 {
        RootsBusy
      } else {
        RootsCloseFailed
      },
    )
  }
  let startup_result = prepare_preflighted(source, limits, prepared, roots)
  match startup_result {
    ReadyProcess(process) =>
      publish_ready_root_bound(ready_owner, failed_owner, process)
    CleanupRequired(child) => {
      let roots_status = close_root_pair_status(roots)
      let roots_live = roots_still_live(roots)
      if !child.is_closed() || roots_live {
        return publish_live_root_bound_cleanup(
          ready_owner, failed_owner, child, roots_live, roots_status,
        )
      }
      ready_owner.roots_selected = false
      failed_owner.roots_selected = false
      failed_owner.child = Some(child)
      failed_owner.roots_cleanup_status = roots_status
      failed_owner.closed = true
      let roots_failure = root_failure_status(roots_status)
      let primary = Process(child.cause())
      let child_cleanup = if child.cleanup_status == 0 {
        None
      } else {
        Some(Process(child.cleanup_failure()))
      }
      match (child_cleanup, roots_failure) {
        (Some(child_failure), Some(root_failure)) =>
          raise RootBoundTripleFailure(primary, child_failure, root_failure)
        (Some(cleanup), None) =>
          raise RootBoundCompoundFailure(primary, cleanup)
        (None, Some(cleanup)) =>
          raise RootBoundCompoundFailure(primary, cleanup)
        (None, None) => raise RootBoundFailure(primary)
      }
    }
  }
}

///|
/// Acquires a reusable rooted supervisor without selecting a transport mode.
pub fn prepare_with_approved_roots(
  executable : Bytes,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  limits : WorkerProcessLimits,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
) -> RootBoundWorkerProcessPreparation raise RootBoundWorkerProcessError {
  prepare_root_bound(
    executable, startup, source, limits, model_root, kernel_root, false, 0UL,
  )
}

///|
/// Production aggregate preparation. Exact predecessor evidence is validated
/// before root acquisition and exchange mode is installed nonfallibly in the
/// preallocated ready owner after startup succeeds.
pub fn prepare_exchange_with_approved_roots(
  executable : Bytes,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  limits : WorkerProcessLimits,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  expected_predecessor : UInt64,
) -> RootBoundWorkerProcessPreparation raise RootBoundWorkerProcessError {
  prepare_root_bound(
    executable, startup, source, limits, model_root, kernel_root, true, expected_predecessor,
  )
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
  let prior_child_failure = self.child_cleanup_failure()
  let prior_roots_failure = if self.roots_cleanup_status != 0 &&
    self.roots_cleanup_status != 7 {
    root_failure_status(self.roots_cleanup_status)
  } else {
    None
  }
  let attempt = attempt_root_bound_cleanup(
    self.child is Some(_),
    self.roots_selected && roots_still_live(self.roots),
    () => {
      guard self.child is Some(child) else { return None }
      try child.retry_close() catch {
        error => Some(Process(error))
      } noraise {
        _ => None
      }
    },
    () => if self.roots_selected { close_root_pair(self.roots) } else { None },
    () => {
      match self.child {
        Some(child) => !child.is_closed()
        None => false
      }
    },
    () => self.roots_selected && roots_still_live(self.roots),
  )
  let child_failure = match attempt.child_failure {
    Some(failure) => Some(failure)
    None => prior_child_failure
  }
  let roots_failure = match attempt.roots_failure {
    Some(failure) => Some(failure)
    None => prior_roots_failure
  }
  self.child_cleanup_process = match child_failure {
    Some(Process(error)) => Some(error)
    _ => None
  }
  self.child_cleanup_native = None
  self.roots_cleanup_status = match roots_failure {
    Some(RootsBusy) => 7
    Some(_) => 1
    None => 0
  }
  self.closed = !attempt.child_live && !attempt.roots_live
  raise_cleanup_attempt({
    child_live: attempt.child_live,
    roots_live: attempt.roots_live,
    child_failure,
    roots_failure,
  })
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
  self.transport_mode != 2 &&
  self.active_exchange_epoch == 0UL &&
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
  if self.transport_mode == 2 {
    raise RootBoundFailure(Process(Invalid(Lifecycle)))
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
  self.transport_mode = 1
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
  if self.transport_mode != 1 {
    raise RootBoundFailure(Process(Invalid(Lifecycle)))
  }
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
