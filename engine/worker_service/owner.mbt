///|
fn scheduler_error(error : @core.SchedulerError) -> WorkerServiceError {
  Scheduler(error)
}

///|
fn WorkerService::require_lifecycle(
  self : WorkerService,
  expected : WorkerServiceLifecycle,
) -> Unit raise WorkerServiceError {
  if self.lifecycle != expected {
    raise Invalid(Lifecycle)
  }
}

///|
fn WorkerService::require_raw_mode(
  self : WorkerService,
) -> Unit raise WorkerServiceError {
  if self.ownership_mode != FixtureRawOwnership {
    raise Invalid(OwnershipMode)
  }
}

///|
fn WorkerService::raw_outstanding_count(self : WorkerService) -> Int {
  let physical = match self.flight {
    FlightEmpty => 0
    _ => 1
  }
  physical + (if self.pending_plan_sequence == 0UL { 0 } else { 1 })
}

///|
fn WorkerService::flight_sequence_value(
  self : WorkerService,
) -> UInt64 raise WorkerServiceError {
  match self.flight {
    FlightEmpty => raise Invalid(Sequence)
    FlightPlanned(sequence)
    | FlightSubmitted(sequence)
    | FlightReceived(sequence)
    | FlightAccepted(sequence)
    | FlightPendingFailure(sequence, _, _) => sequence
  }
}

///|
fn validate_startup_contract(
  scheduler : @core.Scheduler,
  identity : WorkerServiceIdentity,
  binding : WorkerServiceBinding,
  contract : @worker_wire.WorkerStartupContract,
) -> Unit raise WorkerServiceError {
  if contract.identity() != scheduler.model_identity() ||
    contract.identity() != identity.model_identity {
    raise Invalid(Identity)
  }
  if contract.bootstrap_digest() != binding.bootstrap_digest() {
    raise Invalid(BootstrapIdentity)
  }
  if contract.bootstrap_source_digest() != binding.bootstrap_source_digest() {
    raise Invalid(BootstrapSourceIdentity)
  }
  if contract.device_ordinal() != binding.device_ordinal() {
    raise Invalid(DeviceAssignment)
  }
  if contract.model_generation_value() !=
    scheduler.model_plan_generation().value() ||
    contract.model_generation_value() != identity.model_generation_value {
    raise Invalid(Generation)
  }
  if contract.worker_limits() != scheduler.worker_protocol_limits() ||
    contract.worker_limits() != identity.worker_limits {
    raise Invalid(WorkerEnvelope)
  }
  if contract.inference_limits() != binding.inference_limits() ||
    contract.inference_limits() != identity.inference_limits ||
    contract.bootstrap_source_digest() != identity.bootstrap_source_digest {
    raise Invalid(InferenceEnvelope)
  }
  let submitted = scheduler.last_submitted_plan_sequence()
  if scheduler.last_retired_plan_sequence() != submitted ||
    contract.has_predecessor() != (submitted != 0UL) ||
    contract.predecessor_value() != submitted {
    raise Invalid(Sequence)
  }
}

///|
/// Compatibility constructor for deterministic fixtures. The caller retains
/// aliases to both mutable owners, so this service is permanently ineligible
/// for production online-session admission.
pub fn new_fixture(
  scheduler : @core.Scheduler,
  binding : WorkerServiceBinding,
  process : @worker_process.RootBoundWorkerProcessSupervisor,
) -> WorkerService raise WorkerServiceError {
  if process.lifecycle() != RootBoundReady {
    raise Invalid(Lifecycle)
  }
  let identity = worker_service_identity(
    scheduler.model_identity(),
    scheduler.model_plan_generation().value(),
    binding.bootstrap_source_digest(),
    scheduler.worker_protocol_limits(),
    binding.inference_limits(),
  )
  validate_startup_contract(
    scheduler,
    identity,
    binding,
    process.startup_contract(),
  )
  let transport = rooted_single_transport(process)
  transport.claim(scheduler.last_retired_plan_sequence())
  {
    scheduler,
    identity,
    physical_binding: RootedSingleBinding(binding),
    transport,
    flight: FlightEmpty,
    pending_plan_sequence: 0UL,
    maintenance_mode: MAINTENANCE_NONE,
    maintenance_phase: 0,
    maintenance_failure: ExecutionFailed,
    lifecycle: ServiceReady,
    production_owned: false,
    ownership_mode: FixtureRawOwnership,
    online_epoch: 0UL,
    online_publication_cursor: 0UL,
    graph_telemetry: new_worker_service_graph_telemetry(),
  }
}

///|
fn WorkerService::require_transport(
  self : WorkerService,
) -> WorkerPhysicalTransport {
  if self.transport is Uninstalled {
    abort("worker service physical transport installation invariant")
  }
  self.transport
}

///|
/// Returns the immutable backend-neutral identity retained for every
/// replacement.
pub fn WorkerService::identity(self : WorkerService) -> WorkerServiceIdentity {
  self.identity
}

///|
pub fn WorkerService::lifecycle(self : WorkerService) -> WorkerServiceLifecycle {
  self.lifecycle
}

///|
pub fn WorkerService::outstanding_count(
  self : WorkerService,
) -> Int raise WorkerServiceError {
  self.require_raw_mode()
  self.raw_outstanding_count()
}

///|
fn WorkerService::clear_flight(self : WorkerService) -> Unit {
  self.flight = FlightEmpty
}
