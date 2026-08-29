///|
/// Transfers the cleanup owner once. The failed service shell never escapes.
pub fn OwnedWorkerServicePreparation::take_cleanup(
  self : OwnedWorkerServicePreparation,
) -> FailedOwnedWorkerServicePreparation raise OwnedWorkerServicePreparationError {
  if self.consumed || self.state != OwnedServiceCleanupRequired {
    raise owned_failure(OwnedServiceFailure(Invalid(Lifecycle)))
  }
  self.consumed = true
  self.service = None
  self.online = None
  self.cleanup
}

///|
pub fn FailedOwnedWorkerServicePreparation::cause(
  self : FailedOwnedWorkerServicePreparation,
) -> OwnedWorkerServiceFailure {
  match self.cause_kind {
    1 =>
      match self.initial {
        Some(failed) => initial_primary(failed)
        None => abort("owned initial cleanup cause invariant")
      }
    2 =>
      match self.service_cause {
        Some(error) => OwnedServiceFailure(error)
        None => abort("owned service cleanup cause invariant")
      }
    3 =>
      match self.process_cause {
        Some(error) =>
          OwnedPhysicalTransportFailure(physical_transport_failure(error))
        None => abort("owned process cleanup cause invariant")
      }
    _ => abort("owned cleanup cause invariant")
  }
}

///|
pub fn FailedOwnedWorkerServicePreparation::cleanup_failure(
  self : FailedOwnedWorkerServicePreparation,
) -> OwnedWorkerServiceFailure? {
  if self.cleanup_group_failure {
    return Some(OwnedPhysicalTransportFailure(PhysicalCleanupFailed))
  }
  match self.cleanup_process_failure {
    Some(error) =>
      Some(OwnedPhysicalTransportFailure(physical_cleanup_failure(error)))
    None =>
      match self.initial {
        Some(failed) => initial_cleanup_evidence(failed)
        None => None
      }
  }
}

///|
fn progress_failed_group_close(
  group : @tensor_parallel_group_transport.TensorParallelGroupTransport,
) -> Bool raise @tensor_parallel_group_transport.TensorParallelGroupTransportError {
  if group.has_failure_publication_obligation() {
    ignore(take_tensor_parallel_worker_failure(group)) catch {
      PhysicalTransport(_) => ()
      _ => ()
    }
  }
  match group.lifecycle() {
    Closed => return true
    Prepared | Ready => group.begin_close()
    Faulted => group.begin_recovery()
    Recovering | Closing => ()
  }
  if group.lifecycle() == Closed {
    return true
  }
  match group.progress_cleanup() {
    GroupCloseComplete => true
    GroupReplacementReady => {
      group.begin_close()
      group.lifecycle() == Closed
    }
    GroupCleanupAdvanced | GroupCleanupPending => false
    GroupCleanupStuck(_) => raise Rejected(rule=Cleanup, rank=-1)
  }
}

///|
pub fn FailedOwnedWorkerServicePreparation::is_closed(
  self : FailedOwnedWorkerServicePreparation,
) -> Bool {
  self.closed
}

///|
pub fn FailedOwnedWorkerServicePreparation::retry_close(
  self : FailedOwnedWorkerServicePreparation,
) -> Unit raise OwnedWorkerServicePreparationError {
  if self.closed {
    if !self.reported_consumed_failure && self.cleanup_group_failure {
      self.reported_consumed_failure = true
      raise owned_failure(OwnedPhysicalTransportFailure(PhysicalCleanupFailed))
    }
    if !self.reported_consumed_failure &&
      self.cleanup_process_failure is Some(error) {
      self.reported_consumed_failure = true
      raise owned_failure(
        OwnedPhysicalTransportFailure(physical_cleanup_failure(error)),
      )
    }
    return
  }
  self.cleanup_process_failure = None
  self.cleanup_group_failure = false
  if self.initial is Some(failed) {
    try failed.retry_close() catch {
      error => {
        self.cleanup_process_failure = Some(error)
        if !failed.is_closed() {
          raise owned_failure(
            OwnedPhysicalTransportFailure(physical_cleanup_failure(error)),
          )
        }
      }
    } noraise {
      _ => ()
    }
  }
  if self.ready_process is Some(process) {
    try process.close() catch {
      error => {
        self.cleanup_process_failure = Some(error)
        if !process.is_closed() {
          raise owned_failure(
            OwnedPhysicalTransportFailure(physical_cleanup_failure(error)),
          )
        }
      }
    } noraise {
      _ => ()
    }
    if process.is_closed() {
      self.ready_process = None
    }
  }
  if self.group is Some(group) {
    try progress_failed_group_close(group) catch {
      _ => {
        self.cleanup_group_failure = true
        if group.lifecycle() != Closed {
          raise owned_failure(
            OwnedPhysicalTransportFailure(PhysicalCleanupFailed),
          )
        }
      }
    } noraise {
      group_closed => if group_closed { self.group = None }
    }
  }
  let initial_closed = match self.initial {
    Some(failed) => failed.is_closed()
    None => true
  }
  self.closed = initial_closed &&
    self.ready_process is None &&
    self.group is None
  if self.closed && self.cleanup_process_failure is Some(error) {
    self.reported_consumed_failure = true
    raise owned_failure(
      OwnedPhysicalTransportFailure(physical_cleanup_failure(error)),
    )
  }
  if self.closed && self.cleanup_group_failure {
    self.reported_consumed_failure = true
    raise owned_failure(OwnedPhysicalTransportFailure(PhysicalCleanupFailed))
  }
}
