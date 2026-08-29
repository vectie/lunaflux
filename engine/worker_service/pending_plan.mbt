///|
fn WorkerService::has_pending_plan(self : WorkerService) -> Bool {
  self.pending_plan_sequence != 0UL
}

///|
fn WorkerService::stage_pending_plan(
  self : WorkerService,
  sequence : UInt64,
) -> Unit {
  if sequence == 0UL || self.has_pending_plan() {
    abort("worker service pending-plan invariant")
  }
  self.pending_plan_sequence = sequence
}

///|
fn WorkerService::recover_failed_exchange_start(
  self : WorkerService,
  sequence : UInt64,
) -> Unit {
  self.lifecycle = ServiceRecovering
  let submitted = self.require_transport().has_submission(sequence) catch {
      _ => false
    }
  if submitted {
    self.flight = FlightSubmitted(sequence)
  }
}

///|
fn[T] recover_failed_exchange_start_and_raise(
  service : WorkerService,
  sequence : UInt64,
  error : WorkerServiceError,
) -> T raise WorkerServiceError {
  service.recover_failed_exchange_start(sequence)
  raise error
}

///|
fn WorkerService::start_pending_plan(
  self : WorkerService,
) -> WorkerServiceProgress raise WorkerServiceError {
  if !self.has_pending_plan() {
    raise Invalid(Lifecycle)
  }
  let sequence = self.pending_plan_sequence
  let plan = self.scheduler.submitted_plan_for_sequence(sequence) catch {
    error => raise scheduler_error(error)
  }
  if !self.require_transport().has_exchange_capacity() {
    return ServiceBackpressured
  }
  // Transfer the exact scheduler lookup capability into the physical-flight
  // state before the fallible process call. Existing recovery can therefore
  // distinguish an accepted reservation from a never-submitted plan.
  self.flight = FlightPlanned(sequence)
  self.pending_plan_sequence = 0UL
  let actual = self.require_transport().begin_exchange(plan) catch {
      error => recover_failed_exchange_start_and_raise(self, sequence, error)
    }
  if actual != sequence {
    self.lifecycle = ServiceRecovering
    raise Invalid(Sequence)
  }
  self.flight = FlightSubmitted(sequence)
  ServiceStarted(sequence)
}
