///|
pub fn prepare_with_approved_executable(
  executable : @worker_executable_file.WorkerExecutableAdmission,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  limits : WorkerProcessLimits,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  parent_approval? : @worker_wire.WorkerParentApprovalSubjects? = None,
) -> RootBoundWorkerProcessPreparation raise RootBoundWorkerProcessError {
  prepare_root_bound_with_authority(
    executable, startup, source, limits, model_root, kernel_root, false, 0UL, parent_approval,
  )
}

///|
/// Production activation from the exact pinned executable object. This route
/// never accepts, reopens, or rehashes a pathname.
pub fn prepare_exchange_with_approved_executable(
  executable : @worker_executable_file.WorkerExecutableAdmission,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  limits : WorkerProcessLimits,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  expected_predecessor : UInt64,
  parent_approval? : @worker_wire.WorkerParentApprovalSubjects? = None,
) -> RootBoundWorkerProcessPreparation raise RootBoundWorkerProcessError {
  prepare_root_bound_with_authority(
    executable, startup, source, limits, model_root, kernel_root, true, expected_predecessor,
    parent_approval,
  )
}

///|
/// Exhaustively retries both independent initial-startup authorities. A native
/// root close failure that consumed the pair remains evidence, not false retry
/// authority; Busy retains this owner for another attempt.
pub fn FailedRootBoundWorkerProcessStartup::retry_close(
  self : FailedRootBoundWorkerProcessStartup,
) -> Unit raise RootBoundWorkerProcessError {
  if self.closed && !approved_executable_live(self.approved_executable) {
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
  let executable_failure = close_approved_executable(self.approved_executable)
  self.closed = !attempt.child_live &&
    !attempt.roots_live &&
    !approved_executable_live(self.approved_executable)
  if executable_failure is Some(failure) {
    if child_failure is Some(child) {
      if roots_failure is Some(roots) {
        raise RootBoundTripleFailure(child, roots, failure)
      }
      raise RootBoundCompoundFailure(child, failure)
    }
    if roots_failure is Some(roots) {
      raise RootBoundCompoundFailure(roots, failure)
    }
    raise RootBoundFailure(failure)
  }
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
pub fn RootBoundWorkerProcessSupervisor::validate_recovery_binding(
  self : RootBoundWorkerProcessSupervisor,
  identity : @spec.ModelIdentity,
  bootstrap_digest : @worker_wire.WorkerBootstrapDigest,
  bootstrap_source_digest : @worker_wire.WorkerBootstrapSourceDigest,
  device_ordinal : Int,
  model_generation_value : UInt64,
  worker_limits : @worker_protocol.WorkerProtocolLimits,
  inference_limits : @inference.InferenceLimits,
  predecessor_value : UInt64,
) -> Unit raise RootBoundWorkerProcessError {
  if self.lifecycle != RootBoundRecovering {
    raise RootBoundInvalidLifecycle
  }
  let contract = self.process.recovery_startup_contract() catch {
    error => raise RootBoundFailure(Process(error))
  }
  if contract.identity() != identity ||
    contract.bootstrap_digest() != bootstrap_digest ||
    contract.bootstrap_source_digest() != bootstrap_source_digest ||
    contract.device_ordinal() != device_ordinal ||
    contract.model_generation_value() != model_generation_value ||
    contract.worker_limits() != worker_limits ||
    contract.inference_limits() != inference_limits ||
    contract.has_predecessor() != (predecessor_value != 0UL) ||
    contract.predecessor_value() != predecessor_value {
    raise RootBoundOwnerMismatch
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
pub fn RootBoundReceivedWorkerCompletion::graph_telemetry(
  self : RootBoundReceivedWorkerCompletion,
  owner : RootBoundWorkerProcessSupervisor,
) -> @worker_wire.WorkerGraphTelemetry raise RootBoundWorkerProcessError {
  if !physical_equal(self.0, owner) {
    raise RootBoundOwnerMismatch
  }
  ReceivedWorkerCompletion(owner.process, self.1, self.2).graph_telemetry(
    owner.process,
  ) catch {
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
