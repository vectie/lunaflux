///|
fn WorkerService::commit_received(
  self : WorkerService,
) -> UInt64 raise WorkerServiceError {
  let sequence = self.flight_sequence
  let plan = self.scheduler.submitted_plan_for_sequence(sequence) catch {
    error => raise scheduler_error(error)
  }
  let received = self.require_process().received_for_sequence(sequence) catch {
      error => raise root_bound_process_error(error)
    }
  self.require_process().validate_retirement(received) catch {
    error => raise root_bound_process_error(error)
  }
  let frame = received.frame(self.require_process()) catch {
    error => raise root_bound_process_error(error)
  }
  let completion = self.scheduler.accept_completion_frame(plan, frame) catch {
    error => raise scheduler_error(error)
  }
  self.scheduler.complete(plan, completion) catch {
    error => raise scheduler_error(error)
  }
  self.require_process().retire_received(received) catch {
    error => {
      if self.require_process().lifecycle() != RootBoundReady {
        self.lifecycle = ServiceRecovering
      }
      raise root_bound_process_error(error)
    }
  }
  self.clear_flight()
  sequence
}

///|
/// Advances exactly one logical service transition. A completed transport read
/// is pinned and reported before scheduler publication is attempted.
pub fn WorkerService::progress(
  self : WorkerService,
) -> WorkerServiceProgress raise WorkerServiceError {
  self.require_lifecycle(ServiceReady)
  if self.flight_state == FLIGHT_EMPTY {
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
    self.flight_state = FLIGHT_PLANNED
    self.flight_sequence = sequence
    let actual = self.require_process().begin_exchange(plan) catch {
        error => {
          self.lifecycle = ServiceRecovering
          try self.require_process().submission_for_sequence(sequence) catch {
            RootBoundFailure(Process(Invalid(Submission))) => ()
            _ => ()
          } noraise {
            _ => self.flight_state = FLIGHT_SUBMITTED
          }
          raise root_bound_process_error(error)
        }
      }
    if actual != sequence {
      self.lifecycle = ServiceRecovering
      raise Invalid(Sequence)
    }
    self.flight_state = FLIGHT_SUBMITTED
    return ServiceStarted(sequence)
  }
  if self.flight_state == FLIGHT_SUBMITTED {
    let result = self.require_process().progress_exchange() catch {
        error => {
          self.lifecycle = ServiceRecovering
          raise root_bound_process_error(error)
        }
      }
    match result {
      ExchangePending => ServicePending(self.flight_sequence)
      ExchangeCompleted => {
        self.flight_state = FLIGHT_RECEIVED
        ServiceCompletionReady(self.flight_sequence)
      }
    }
  } else if self.flight_state == FLIGHT_RECEIVED {
    ServiceCommitted(self.commit_received())
  } else {
    raise Invalid(Lifecycle)
  }
}
