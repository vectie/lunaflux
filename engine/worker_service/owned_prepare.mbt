///|
fn owned_failure(
  failure : OwnedWorkerServiceFailure,
) -> OwnedWorkerServicePreparationError {
  OwnedPreparationFailure(failure)
}

///|
fn initial_primary(
  failed : @worker_process.FailedRootBoundWorkerProcessStartup,
) -> OwnedWorkerServiceFailure {
  OwnedProcessFailure(RootBoundFailure(failed.cause()))
}

///|
fn initial_cleanup_evidence(
  failed : @worker_process.FailedRootBoundWorkerProcessStartup,
) -> OwnedWorkerServiceFailure? {
  match (failed.child_cleanup_failure(), failed.roots_cleanup_failure()) {
    (Some(child), Some(roots)) =>
      Some(OwnedProcessFailure(RootBoundCompoundFailure(child, roots)))
    (Some(child), None) => Some(OwnedProcessFailure(RootBoundFailure(child)))
    (None, Some(roots)) => Some(OwnedProcessFailure(RootBoundFailure(roots)))
    (None, None) => None
  }
}

///|
fn validate_owned_preflight(
  blueprint : @core.SchedulerBlueprint,
  binding : WorkerServiceBinding,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
) -> Unit raise OwnedWorkerServicePreparationError {
  if source.digest() != startup.bootstrap_source_digest() {
    raise owned_failure(OwnedServiceFailure(Invalid(BootstrapSourceIdentity)))
  }
  if startup.identity() != blueprint.model_identity() {
    raise owned_failure(OwnedServiceFailure(Invalid(Identity)))
  }
  if startup.model_generation_value() !=
    blueprint.model_plan_generation().value() {
    raise owned_failure(OwnedServiceFailure(Invalid(Generation)))
  }
  if startup.worker_limits() != blueprint.worker_protocol_limits() {
    raise owned_failure(OwnedServiceFailure(Invalid(WorkerEnvelope)))
  }
  if startup.bootstrap_digest() != binding.bootstrap_digest() {
    raise owned_failure(OwnedServiceFailure(Invalid(BootstrapIdentity)))
  }
  if startup.bootstrap_source_digest() != binding.bootstrap_source_digest() {
    raise owned_failure(OwnedServiceFailure(Invalid(BootstrapSourceIdentity)))
  }
  if startup.device_ordinal() != binding.device_ordinal() {
    raise owned_failure(OwnedServiceFailure(Invalid(DeviceAssignment)))
  }
  if startup.inference_limits() != binding.inference_limits() {
    raise owned_failure(OwnedServiceFailure(Invalid(InferenceEnvelope)))
  }
  if startup.has_predecessor() || startup.predecessor_value() != 0UL {
    raise owned_failure(OwnedServiceFailure(Invalid(Sequence)))
  }
}

///|
fn publish_cleanup(
  prepared : OwnedWorkerServicePreparation,
) -> OwnedWorkerServicePreparation {
  prepared.state = OwnedServiceCleanupRequired
  prepared
}

///|
fn install_owned_ready(
  prepared : OwnedWorkerServicePreparation,
  service : WorkerService,
  process : @worker_process.RootBoundWorkerProcessSupervisor,
) -> OwnedWorkerServicePreparation {
  service.process = Some(process)
  prepared.state = OwnedServiceReady
  prepared
}

///|
/// Constructs both mutable owners internally. The ordinary roots are borrowed;
/// the returned preparation shell retains either the service or the only
/// remaining cleanup authority. The shell and both possible owners are
/// allocated before root acquisition.
pub fn prepare_owned(
  blueprint : @core.SchedulerBlueprint,
  binding : WorkerServiceBinding,
  executable : Bytes,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  process_limits : @worker_process.WorkerProcessLimits,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
) -> OwnedWorkerServicePreparation raise OwnedWorkerServicePreparationError {
  validate_owned_preflight(blueprint, binding, startup, source)
  let scheduler = blueprint.instantiate() catch {
    error => raise owned_failure(OwnedSchedulerFailure(error))
  }
  let service : WorkerService = {
    scheduler,
    binding,
    process: None,
    flight_state: FLIGHT_EMPTY,
    flight_sequence: 0UL,
    lifecycle: ServiceReady,
    production_owned: true,
    ownership_mode: SERVICE_MODE_ONLINE_ELIGIBLE,
    online_epoch: 0UL,
    online_publication_cursor: 0UL,
  }
  let failed_owner : FailedOwnedWorkerServicePreparation = {
    initial: None,
    ready_process: None,
    cause_kind: 0,
    service_cause: None,
    process_cause: None,
    cleanup_process_failure: None,
    closed: false,
    reported_consumed_failure: false,
  }
  let outcome : OwnedWorkerServicePreparation = {
    service: Some(service),
    cleanup: failed_owner,
    state: OwnedServiceCleanupRequired,
    consumed: false,
  }
  let rooted = @worker_process.prepare_exchange_with_approved_roots(
    executable,
    startup,
    source,
    process_limits,
    model_root,
    kernel_root,
    scheduler.last_retired_plan_sequence(),
  ) catch {
    error => raise owned_failure(OwnedProcessFailure(error))
  }
  match rooted {
    RootBoundCleanupRequired(failed) => {
      if failed.is_closed() {
        let primary = initial_primary(failed)
        match initial_cleanup_evidence(failed) {
          Some(cleanup) =>
            raise OwnedPreparationCompoundFailure(primary, cleanup)
          None => raise OwnedPreparationFailure(primary)
        }
      }
      failed_owner.initial = Some(failed)
      failed_owner.cause_kind = 1
      publish_cleanup(outcome)
    }
    ReadyRootBoundProcess(process) => {
      service.process = Some(process)
      if !service.production_owned {
        abort("owned service provenance invariant")
      }
      install_owned_ready(outcome, service, process)
    }
  }
}

///|
pub fn OwnedWorkerServicePreparation::state(
  self : OwnedWorkerServicePreparation,
) -> OwnedWorkerServicePreparationState {
  self.state
}

///|
/// Transfers the ready service once. No cleanup authority can be extracted
/// from the same aliased shell afterward.
pub fn OwnedWorkerServicePreparation::take_ready(
  self : OwnedWorkerServicePreparation,
) -> WorkerService raise OwnedWorkerServicePreparationError {
  if self.consumed || self.state != OwnedServiceReady {
    raise owned_failure(OwnedServiceFailure(Invalid(Lifecycle)))
  }
  self.consumed = true
  match self.service {
    Some(service) => {
      self.service = None
      service
    }
    None => abort("owned ready service invariant")
  }
}

///|
/// Transfers the cleanup owner once. The failed service shell never escapes.
pub fn OwnedWorkerServicePreparation::take_cleanup(
  self : OwnedWorkerServicePreparation,
) -> FailedOwnedWorkerServicePreparation raise OwnedWorkerServicePreparationError {
  if self.consumed || self.state != OwnedServiceCleanupRequired {
    raise owned_failure(OwnedServiceFailure(Invalid(Lifecycle)))
  }
  self.consumed = true
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
        Some(error) => OwnedProcessFailure(error)
        None => abort("owned process cleanup cause invariant")
      }
    _ => abort("owned cleanup cause invariant")
  }
}

///|
pub fn FailedOwnedWorkerServicePreparation::cleanup_failure(
  self : FailedOwnedWorkerServicePreparation,
) -> OwnedWorkerServiceFailure? {
  match self.cleanup_process_failure {
    Some(error) => Some(OwnedProcessFailure(error))
    None =>
      match self.initial {
        Some(failed) => initial_cleanup_evidence(failed)
        None => None
      }
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
    if !self.reported_consumed_failure &&
      self.cleanup_process_failure is Some(error) {
      self.reported_consumed_failure = true
      raise owned_failure(OwnedProcessFailure(error))
    }
    return
  }
  self.cleanup_process_failure = None
  if self.initial is Some(failed) {
    try failed.retry_close() catch {
      error => {
        self.cleanup_process_failure = Some(error)
        if !failed.is_closed() {
          raise owned_failure(OwnedProcessFailure(error))
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
          raise owned_failure(OwnedProcessFailure(error))
        }
      }
    } noraise {
      _ => ()
    }
    if process.is_closed() {
      self.ready_process = None
    }
  }
  let initial_closed = match self.initial {
    Some(failed) => failed.is_closed()
    None => true
  }
  self.closed = initial_closed && self.ready_process is None
  if self.closed && self.cleanup_process_failure is Some(error) {
    self.reported_consumed_failure = true
    raise owned_failure(OwnedProcessFailure(error))
  }
}
