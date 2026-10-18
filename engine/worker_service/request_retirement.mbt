///|
fn WorkerPhysicalTransport::requires_request_retirement(self : Self) -> Bool {
  match self {
    RootedSingle(process) => process.requires_request_retirement()
    _ => false
  }
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
