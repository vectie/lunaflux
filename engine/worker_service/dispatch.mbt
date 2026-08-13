///|
pub fn WorkerService::submit_next(
  self : WorkerService,
) -> WorkerServiceDispatch raise WorkerServiceError {
  self.require_lifecycle(ServiceReady)
  let slot = self.empty_slot()
  if slot < 0 {
    return DispatchBackpressured
  }
  let outcome = self.scheduler.build_next() catch {
    error => raise scheduler_error(error)
  }
  if outcome.is_idle() {
    return DispatchIdle
  }
  if outcome.is_backpressured() {
    return DispatchBackpressured
  }
  let plan = self.scheduler.submitted_plan(outcome) catch {
    error => raise scheduler_error(error)
  }
  let sequence = plan.sequence_wire() catch {
    error => raise Scheduler(WorkerProtocol(error))
  }
  self.flight_states[slot] = FLIGHT_PLANNED
  self.flight_sequences[slot] = sequence
  ignore(
    self.process.submit(plan) catch {
      error => {
        if self.process.lifecycle() != RootBoundReady {
          self.lifecycle = ServiceRecovering
        }
        raise root_bound_process_error(error)
      }
    },
  )
  self.flight_states[slot] = FLIGHT_SUBMITTED
  DispatchSubmitted(sequence)
}

///|
fn WorkerService::commit_received_slot(
  self : WorkerService,
  slot : Int,
) -> UInt64 raise WorkerServiceError {
  let sequence = self.flight_sequences[slot]
  let plan = self.scheduler.submitted_plan_for_sequence(sequence) catch {
    error => raise scheduler_error(error)
  }
  let received = self.process.received_for_sequence(sequence) catch {
    error => raise root_bound_process_error(error)
  }
  self.process.validate_retirement(received) catch {
    error => raise root_bound_process_error(error)
  }
  let frame = received.frame(self.process) catch {
    error => raise root_bound_process_error(error)
  }
  let completion = self.scheduler.accept_completion_frame(plan, frame) catch {
    error => raise scheduler_error(error)
  }
  self.scheduler.complete(plan, completion) catch {
    error => raise scheduler_error(error)
  }
  self.process.retire_received(received) catch {
    error => {
      if self.process.lifecycle() != RootBoundReady {
        self.lifecycle = ServiceRecovering
      }
      raise root_bound_process_error(error)
    }
  }
  self.clear_slot(slot)
  sequence
}

///|
/// Receives and commits the oldest process result. Scheduler publication
/// backpressure leaves the exact received frame pinned for retry.
pub fn WorkerService::complete_oldest(
  self : WorkerService,
) -> UInt64 raise WorkerServiceError {
  self.require_lifecycle(ServiceReady)
  let slot = self.oldest_slot()
  if slot < 0 {
    raise Invalid(Sequence)
  }
  let sequence = self.flight_sequences[slot]
  if self.flight_states[slot] == FLIGHT_SUBMITTED {
    let plan = self.scheduler.submitted_plan_for_sequence(sequence) catch {
      error => raise scheduler_error(error)
    }
    let submission = self.process.submission_for_sequence(sequence) catch {
      error => raise root_bound_process_error(error)
    }
    ignore(
      self.process.receive(submission, plan) catch {
        error => {
          if self.process.lifecycle() != RootBoundReady {
            self.lifecycle = ServiceRecovering
          }
          raise root_bound_process_error(error)
        }
      },
    )
    self.flight_states[slot] = FLIGHT_RECEIVED
  }
  if self.flight_states[slot] != FLIGHT_RECEIVED {
    raise Invalid(Sequence)
  }
  self.commit_received_slot(slot)
}
