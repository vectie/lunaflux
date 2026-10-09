///|
/// Control-plane handle. The online service retains the same owner; callers
/// drive authenticated sessions and supply release receipts/replacement plans.
pub struct RemoteWorkerServiceControl {
  priv mut current : @remote_group.RemoteGroup
  priv mut replacement : @remote_group.RemoteGroup?
  priv mut replacement_cleanup : Bool
  priv mut generation_floor : UInt64
  priv identities : FixedArray[@worker_protocol.PlanBufferIdentity]
  priv limits : @remote_wire.RankGroupWireLimits
  priv timeout : UInt64
  priv identity : WorkerServiceIdentity
  priv mut previous : UInt64
  priv mut recovering : Bool
  priv mut failure_taken : Bool
  priv mut closed : Bool
}

///|
pub fn RemoteWorkerServiceControl::group(
  self : RemoteWorkerServiceControl,
) -> @remote_group.RemoteGroup {
  self.current
}

///|
fn remote_error(_error : Error) -> WorkerServiceError {
  PhysicalTransport(PhysicalUnavailable)
}

///|
fn RemoteWorkerServiceControl::failure(
  self : RemoteWorkerServiceControl,
) -> @worker_protocol.WorkerFailure raise WorkerServiceError {
  let report = if self.failure_taken {
    self.current.taken_failure() catch {
      error => raise remote_error(error)
    }
  } else {
    let report = self.current.take_failure() catch {
      error => raise remote_error(error)
    }
    self.failure_taken = true
    report
  }
  tensor_parallel_worker_failure(report)
}

///|
fn RemoteWorkerServiceControl::begin_recovery(
  self : RemoteWorkerServiceControl,
) -> Unit raise WorkerServiceError {
  if self.current.lifecycle() != Faulted {
    raise Invalid(Recovery)
  }
  ignore(self.failure())
  self.recovering = true
}

///|
pub fn RemoteWorkerServiceControl::confirm_release(
  self : RemoteWorkerServiceControl,
  rank : Int,
  receipt : @remote_release.RemoteReleaseReceipt,
) -> Unit raise {
  self.current.confirm_release(rank, receipt)
}

///|
fn RemoteWorkerServiceControl::retire_failed(
  self : RemoteWorkerServiceControl,
  sequence : UInt64,
) -> Unit raise WorkerServiceError {
  if !self.recovering ||
    !self.current.recovery_cleanup_complete() ||
    sequence != self.previous + 1UL {
    raise Invalid(Recovery)
  }
  self.previous = sequence
}

///|
fn RemoteWorkerServiceControl::validate_identity(
  self : RemoteWorkerServiceControl,
  startup : @worker_wire.WorkerStartupContract,
) -> Unit raise WorkerServiceError {
  if startup.identity() != self.identity.model_identity ||
    startup.model_generation_value() != self.identity.model_generation_value ||
    startup.bootstrap_source_digest() != self.identity.bootstrap_source_digest ||
    startup.worker_limits() != self.identity.worker_limits ||
    startup.inference_limits() != self.identity.inference_limits ||
    startup.predecessor_value() != self.previous {
    raise Invalid(Identity)
  }
}

///|
/// Prepares a complete fresh group only after both old ranks have authenticated
/// release receipts and scheduler retirement has advanced the predecessor.
pub fn RemoteWorkerServiceControl::prepare_replacement(
  self : RemoteWorkerServiceControl,
  remote : @remote_startup.RemoteGroupStartup,
  admitted : @remote_bootstrap.TensorParallelGroupBootstrap,
) -> @remote_group.RemoteGroup raise {
  if !self.recovering ||
    !self.current.recovery_cleanup_complete() ||
    self.replacement is Some(_) ||
    remote.rank(0).unwrap().group_generation_value() <= self.generation_floor ||
    remote.digest_sha256() == self.current.startup().digest_sha256() {
    raise Invalid(Recovery)
  }
  self.validate_identity(remote.rank(0).unwrap().worker())
  self.validate_identity(remote.rank(1).unwrap().worker())
  let replacement = @remote_group.RemoteGroup::new(
    remote,
    admitted,
    self.identities,
    self.limits,
    timeout_millis=self.timeout,
  )
  self.replacement = Some(replacement)
  self.generation_floor = replacement.generation()
  replacement
}

///|
/// A receipt for the old group never retires a failed replacement's resources.
pub fn RemoteWorkerServiceControl::confirm_replacement_release(
  self : RemoteWorkerServiceControl,
  rank : Int,
  receipt : @remote_release.RemoteReleaseReceipt,
) -> Unit raise {
  guard self.replacement is Some(replacement) else { raise Invalid(Recovery) }
  replacement.confirm_release(rank, receipt)
}

///|
fn RemoteWorkerServiceControl::finish_replacement_cleanup(
  self : RemoteWorkerServiceControl,
) -> Bool raise WorkerServiceError {
  guard self.replacement is Some(replacement) else { raise Invalid(Recovery) }
  if !self.replacement_cleanup || replacement.lifecycle() != Faulted {
    raise Invalid(Recovery)
  }
  if !replacement.recovery_cleanup_complete() {
    return false
  }
  self.replacement = None
  self.replacement_cleanup = false
  true
}

///|
fn RemoteWorkerServiceControl::restart(
  self : RemoteWorkerServiceControl,
) -> WorkerPhysicalRestart raise WorkerServiceError {
  guard self.replacement is Some(replacement) else {
    raise PhysicalTransport(PhysicalBackpressured)
  }
  if replacement.lifecycle() == Faulted {
    self.replacement_cleanup = true
    return PhysicalRestartCleanupRequired
  }
  if replacement.lifecycle() != Ready ||
    !self.current.recovery_cleanup_complete() {
    raise PhysicalTransport(PhysicalBackpressured)
  }
  self.validate_identity(replacement.startup().rank(0).unwrap().worker())
  self.validate_identity(replacement.startup().rank(1).unwrap().worker())
  self.current = replacement
  self.replacement = None
  self.recovering = false
  self.failure_taken = false
  PhysicalRestarted
}

///|
fn RemoteWorkerServiceControl::begin_close(
  self : RemoteWorkerServiceControl,
) -> Unit raise WorkerServiceError {
  if self.closed {
    return
  }
  if self.replacement is Some(_) {
    raise PhysicalTransport(PhysicalBackpressured)
  }
  match self.current.lifecycle() {
    Ready =>
      self.current.begin_close() catch {
        error => raise remote_error(error)
      }
    Closed => self.closed = true
    Faulted =>
      if self.current.recovery_cleanup_complete() {
        self.closed = true
      } else {
        raise PhysicalTransport(PhysicalCleanupFailed)
      }
    Closing => ()
    _ => raise Invalid(Lifecycle)
  }
}

///|
fn RemoteWorkerServiceControl::maintenance(
  self : RemoteWorkerServiceControl,
) -> WorkerPhysicalMaintenanceProgress raise WorkerServiceError {
  if self.closed {
    return WorkerPhysicalMaintenanceProgress(5)
  }
  if self.replacement_cleanup {
    return WorkerPhysicalMaintenanceProgress(
      if self.finish_replacement_cleanup() {
        5
      } else {
        2
      },
    )
  }
  if self.recovering {
    return WorkerPhysicalMaintenanceProgress(
      if self.current.recovery_cleanup_complete() {
        5
      } else {
        2
      },
    )
  }
  if self.current.lifecycle() == Closing {
    let complete = self.current.progress_close() catch {
      error => raise remote_error(error)
    }
    if complete {
      self.closed = true
      return WorkerPhysicalMaintenanceProgress(5)
    }
    return WorkerPhysicalMaintenanceProgress(3)
  }
  raise Invalid(Lifecycle)
}
