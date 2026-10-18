///|
fn WorkerPhysicalTransport::requires_request_retirement(self : Self) -> Bool {
  match self {
    RootedSingle(process) => process.requires_request_retirement()
    _ => false
  }
}

///|
/// Preserve the scheduler's immutable eviction effect until both device
/// ranks acknowledge it. The logical request remains live for recomputation.
fn WorkerService::retain_preemption(
  self : Self,
  outcome : @core.BuildNextOutcome,
) -> Unit raise WorkerServiceError {
  if !self.transport.requires_request_retirement() {
    return
  }
  guard self.flight is FlightEmpty && !self.has_pending_plan() else {
    raise Invalid(Lifecycle)
  }
  self.flight = FlightEvicting(
    outcome.preempted_request_id() catch {
      e => raise scheduler_error(e)
    },
    outcome.preempted_request_generation() catch {
      e => raise scheduler_error(e)
    },
  )
}

///|
/// Device state is no longer needed after terminal publication. Retire it
/// before the next plan, independently of a slow HTTP reader's event ACK.
fn OnlineWorkerLease::progress_retired_lanes(
  self : Self,
) -> Bool raise WorkerServiceError {
  if !self.service.transport.requires_request_retirement() {
    return true
  }
  let lane = self.next_device_retirement_lane()
  if lane >= 0 {
    let request : LunaOnlineWorkerRequest = {
      lease: self,
      lease_epoch: self.epoch,
      lane,
      generation: self.lane_generations[lane],
    }
    ignore(request.progress_terminal_retirement())
    return false
  }
  true
}

///|
fn OnlineWorkerLease::next_device_retirement_lane(self : Self) -> Int {
  if self.device_retirement_lane >= 0 {
    return self.device_retirement_lane
  }
  for lane in 0..<self.lane_phases.length() {
    if self.lane_phases[lane] == ONLINE_LANE_TERMINAL &&
      !self.lane_device_retired[lane] {
      return lane
    }
  }
  -1
}

///|
/// Release the physical request before the HTTP lane/claim can be reused.
/// A pending step is progressed without constructing a new plan. The original
/// generation is retained across cancellation; a cancellation generation is
/// never substituted for the generation that owns device state.
pub fn LunaOnlineWorkerRequest::progress_terminal_retirement(
  self : Self,
) -> Bool raise WorkerServiceError {
  let lease = self.require_lane()
  let service = lease.service
  guard lease.lane_phases[self.lane] == ONLINE_LANE_TERMINAL &&
    lease.cancel_lane < 0 &&
    service.lifecycle == ServiceReady else {
    raise Invalid(Lifecycle)
  }
  if !service.transport.requires_request_retirement() ||
    lease.lane_device_retired[self.lane] {
    return true
  }
  if service.raw_outstanding_count() != 0 {
    ignore(service.progress_existing_flight_impl())
    return false
  }
  if lease.device_retirement_lane >= 0 &&
    lease.device_retirement_lane != self.lane {
    return false
  }
  guard service.transport is RootedSingle(process) &&
    lease.lane_handles[self.lane].length() == 1 else {
    raise Invalid(Lifecycle)
  }
  lease.device_retirement_lane = self.lane
  let handle = lease.lane_handles[self.lane][0]
  let done = process.progress_request_retirement(
    handle.request_id().value(),
    lease.lane_cut_source_generations[self.lane].value(),
  ) catch {
    e => raise physical_transport_error(e)
  }
  if done {
    lease.lane_device_retired[self.lane] = true
    lease.device_retirement_lane = -1
  }
  done
}
