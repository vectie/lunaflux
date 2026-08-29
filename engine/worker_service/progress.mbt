///|
fn WorkerService::commit_received(
  self : WorkerService,
) -> WorkerServiceProgress raise WorkerServiceError {
  let sequence = self.flight_sequence_value()
  let plan = self.scheduler.submitted_plan_for_sequence(sequence) catch {
    error => raise scheduler_error(error)
  }
  let commit = match self.flight {
    FlightReceived(_) => {
      let frame = self.require_transport().received_frame(sequence)
      let completion = bridge_completion(self.scheduler, plan, frame)
      self.require_transport().accept_completion(sequence, plan, completion)
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
  self.require_transport().retire_accepted(sequence)
  self.clear_flight()
  ServiceCommitted(sequence)
}

///|
fn[T] enter_service_recovery(
  service : WorkerService,
  error : WorkerServiceError,
) -> T raise WorkerServiceError {
  service.lifecycle = ServiceRecovering
  raise error
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
      self.require_transport().probe_idle_health() catch {
        error => enter_service_recovery(self, error)
      }
      if self.has_pending_plan() {
        return self.start_pending_plan()
      }
      if !self.require_transport().has_exchange_capacity() {
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
      if outcome.is_preempted() {
        return ServiceAdvanced
      }
      let plan = self.scheduler.submitted_plan(outcome) catch {
        error => raise scheduler_error(error)
      }
      let sequence = plan.sequence_wire() catch {
        error => raise Scheduler(WorkerProtocol(error))
      }
      self.stage_pending_plan(sequence)
      self.start_pending_plan()
    }
    FlightSubmitted(_) => {
      if !self.has_pending_plan() {
        let outcome = self.scheduler.build_next() catch {
          error => raise scheduler_error(error)
        }
        if outcome.is_preempted() {
          return ServiceAdvanced
        }
        if outcome.is_submitted() {
          let plan = self.scheduler.submitted_plan(outcome) catch {
            error => raise scheduler_error(error)
          }
          let sequence = plan.sequence_wire() catch {
            error => raise Scheduler(WorkerProtocol(error))
          }
          self.stage_pending_plan(sequence)
          return ServicePlanPending(sequence)
        }
      }
      self.progress_existing_flight_impl()
    }
    FlightReceived(_) | FlightAccepted(_) => self.commit_received()
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
      let plan = self.scheduler.submitted_plan_for_sequence(sequence) catch {
        error => raise scheduler_error(error)
      }
      let result = self.require_transport().progress_exchange(plan) catch {
          error => enter_service_recovery(self, error)
        }
      match result {
        PhysicalExchangePending => ServicePending(sequence)
        PhysicalExchangeCompleted => {
          let transport = self.require_transport()
          if transport.supports_graph_telemetry() {
            let graph_report = transport.received_graph_telemetry(sequence) catch {
              error => enter_service_recovery(self, error)
            }
            self.graph_telemetry.record_child_report(sequence, graph_report) catch {
              error => enter_service_recovery(self, error)
            }
          }
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
