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
  {
    scheduler,
    binding,
    process,
    flight_states: FixedArray::make(2, FLIGHT_EMPTY),
    flight_sequences: FixedArray::make(2, 0UL),
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
  (if self.flight_states[0] == FLIGHT_EMPTY { 0 } else { 1 }) +
  (if self.flight_states[1] == FLIGHT_EMPTY { 0 } else { 1 })
}

///|
fn WorkerService::empty_slot(self : WorkerService) -> Int {
  if self.flight_states[0] == FLIGHT_EMPTY {
    0
  } else if self.flight_states[1] == FLIGHT_EMPTY {
    1
  } else {
    -1
  }
}

///|
fn WorkerService::oldest_slot(self : WorkerService) -> Int {
  let state_a = self.flight_states[0]
  let state_b = self.flight_states[1]
  if state_a != FLIGHT_EMPTY &&
    (
      state_b == FLIGHT_EMPTY ||
      self.flight_sequences[0] < self.flight_sequences[1]
    ) {
    0
  } else if state_b != FLIGHT_EMPTY {
    1
  } else {
    -1
  }
}

///|
fn WorkerService::clear_slot(self : WorkerService, slot : Int) -> Unit {
  self.flight_states[slot] = FLIGHT_EMPTY
  self.flight_sequences[slot] = 0UL
}
