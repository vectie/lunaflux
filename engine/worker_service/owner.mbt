///|
fn process_error(
  error : @worker_process.WorkerProcessError,
) -> WorkerServiceError {
  Process(error)
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
fn validate_startup_against_scheduler(
  scheduler : @core.Scheduler,
  startup : @worker_process.WorkerProcessSupervisor,
) -> Unit raise WorkerServiceError {
  let contract = startup.startup_contract()
  if contract.identity() != scheduler.model_identity() {
    raise Invalid(Identity)
  }
  if contract.model_generation_value() !=
    scheduler.model_plan_generation().value() {
    raise Invalid(Generation)
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
  process : @worker_process.WorkerProcessSupervisor,
) -> WorkerService raise WorkerServiceError {
  if process.lifecycle() != Ready {
    raise Invalid(Lifecycle)
  }
  validate_startup_against_scheduler(scheduler, process)
  {
    scheduler,
    process,
    flight_states: FixedArray::make(2, FLIGHT_EMPTY),
    flight_sequences: FixedArray::make(2, 0UL),
    lifecycle: ServiceReady,
  }
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
