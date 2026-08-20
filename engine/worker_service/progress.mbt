///|
fn WorkerService::commit_received(
  self : WorkerService,
) -> WorkerServiceProgress raise WorkerServiceError {
  let sequence = self.flight_sequence_value()
  let plan = self.scheduler.submitted_plan_for_sequence(sequence) catch {
    error => raise scheduler_error(error)
  }
  let received = self.require_process().received_for_sequence(sequence) catch {
      error => raise root_bound_process_error(error)
    }
  self.require_process().validate_retirement(received) catch {
    error => raise root_bound_process_error(error)
  }
  let commit = match self.flight {
    FlightReceived(_) => {
      let frame = received.frame(self.require_process()) catch {
        error => raise root_bound_process_error(error)
      }
      let completion = bridge_completion(self.scheduler, plan, frame)
      self.flight = FlightAccepted(sequence)
      self.scheduler.complete(plan, completion) catch {
        error => raise scheduler_error(error)
      }
    }
    FlightAccepted(_) =>
      self.scheduler.complete_submitted(plan) catch {
        error => raise scheduler_error(error)
      }
    _ => raise Invalid(Lifecycle)
  }
  if commit is CompletionBackpressured {
    return ServiceBackpressured
  }
  try! self.require_process().retire_received(received)
  self.clear_flight()
  ServiceCommitted(sequence)
}

///|
/// Advances exactly one logical service transition. A completed transport read
/// is pinned and reported before scheduler publication is attempted.
fn WorkerService::progress_impl(
  self : WorkerService,
) -> WorkerServiceProgress raise WorkerServiceError {
  self.require_lifecycle(ServiceReady)
  match self.flight {
    FlightEmpty => {
      if !self.require_process().has_exchange_capacity() {
        return ServiceBackpressured
      }
      let outcome = self.scheduler.build_next() catch {
        error => raise scheduler_error(error)
      }
      if outcome.is_idle() {
        return ServiceIdle
      }
      if outcome.is_backpressured() {
        return ServiceBackpressured
      }
      let plan = self.scheduler.submitted_plan(outcome) catch {
        error => raise scheduler_error(error)
      }
      let sequence = plan.sequence_wire() catch {
        error => raise Scheduler(WorkerProtocol(error))
      }
      self.flight = FlightPlanned(sequence)
      let actual = self.require_process().begin_exchange(plan) catch {
          error => {
            self.lifecycle = ServiceRecovering
            try self.require_process().submission_for_sequence(sequence) catch {
              RootBoundFailure(Process(Invalid(Submission))) => ()
              _ => ()
            } noraise {
              _ => self.flight = FlightSubmitted(sequence)
            }
            raise root_bound_process_error(error)
          }
        }
      if actual != sequence {
        self.lifecycle = ServiceRecovering
        raise Invalid(Sequence)
      }
      self.flight = FlightSubmitted(sequence)
      ServiceStarted(sequence)
    }
    _ => self.progress_existing_flight_impl()
  }
}

///|
/// Advances only an already-recorded physical exchange. This never asks the
/// scheduler to build another plan and is therefore safe after a logical
/// cancel/deadline/terminal cut still needs to retire stale device work.
fn WorkerService::progress_existing_flight_impl(
  self : WorkerService,
) -> WorkerServiceProgress raise WorkerServiceError {
  self.require_lifecycle(ServiceReady)
  match self.flight {
    FlightSubmitted(sequence) => {
      let result = self.require_process().progress_exchange() catch {
          error => {
            self.lifecycle = ServiceRecovering
            raise root_bound_process_error(error)
          }
        }
      match result {
        ExchangePending => ServicePending(sequence)
        ExchangeCompleted => {
          self.flight = FlightReceived(sequence)
          ServiceCompletionReady(sequence)
        }
      }
    }
    FlightReceived(_) | FlightAccepted(_) => self.commit_received()
    _ => raise Invalid(Lifecycle)
  }
}

///|
pub fn WorkerService::progress(
  self : WorkerService,
) -> WorkerServiceProgress raise WorkerServiceError {
  self.require_raw_mode()
  self.progress_impl()
}
