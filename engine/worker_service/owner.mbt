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
pub fn new(
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
    process,
    flight_state: FLIGHT_EMPTY,
    flight_sequence: 0UL,
    lifecycle: ServiceReady,
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
pub fn WorkerService::outstanding_count(self : WorkerService) -> Int {
  if self.flight_state == FLIGHT_EMPTY {
    0
  } else {
    1
  }
}

///|
fn WorkerService::clear_flight(self : WorkerService) -> Unit {
  self.flight_state = FLIGHT_EMPTY
  self.flight_sequence = 0UL
}
