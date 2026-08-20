///|
fn root_bound_process_error(
  error : @worker_process.RootBoundWorkerProcessError,
) -> WorkerServiceError {
  RootBoundProcess(error)
}

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
  match self.flight {
    FlightEmpty => 0
    _ => 1
  }
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
    | FlightAccepted(sequence) => sequence
  }
}

///|
fn validate_startup_contract(
  scheduler : @core.Scheduler,
  binding : WorkerServiceBinding,
  contract : @worker_wire.WorkerStartupContract,
) -> Unit raise WorkerServiceError {
  if contract.identity() != scheduler.model_identity() {
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
    scheduler.model_plan_generation().value() {
    raise Invalid(Generation)
  }
  if contract.worker_limits() != scheduler.worker_protocol_limits() {
    raise Invalid(WorkerEnvelope)
  }
  if contract.inference_limits() != binding.inference_limits() {
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
  validate_startup_contract(scheduler, binding, process.startup_contract())
  process.claim_exchange(scheduler.last_retired_plan_sequence()) catch {
    error => raise root_bound_process_error(error)
  }
  {
    scheduler,
    binding,
    process: Some(process),
    flight: FlightEmpty,
    lifecycle: ServiceReady,
    production_owned: false,
    ownership_mode: FixtureRawOwnership,
    online_epoch: 0UL,
    online_publication_cursor: 0UL,
  }
}

///|
fn WorkerService::require_process(
  self : WorkerService,
) -> @worker_process.RootBoundWorkerProcessSupervisor {
  match self.process {
    Some(process) => process
    None => abort("worker service process installation invariant")
  }
}

///|
/// Returns the immutable deployment binding retained for every replacement.
pub fn WorkerService::binding(self : WorkerService) -> WorkerServiceBinding {
  self.binding
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
