///|
/// Samples the retained monotonic clock immediately before scheduler mutation.
pub fn OnlineWorkerLease::admit(
  self : OnlineWorkerLease,
  request : @core.TokenizedRequest,
) -> Unit raise WorkerServiceError {
  if self.active_lane_count != 0 || self.compatibility_lane >= 0 {
    raise Invalid(Lifecycle)
  }
  let admitted = self.try_admit_request(request)
  match admitted.kind() {
    LunaOnlineWorkerAdmitted => {
      let exact = try! admitted.request()
      self.compatibility_lane = exact.lane
      self.compatibility_generation = exact.generation
    }
    LunaOnlineWorkerSaturated | LunaOnlineWorkerDraining =>
      raise Invalid(Lifecycle)
  }
}

///|
/// Commits the exact shell preallocated before rooted worker activation.
pub fn OnlineWorkerLease::commit_prepared_admission(
  self : OnlineWorkerLease,
) -> OnlinePreparedAdmissionCommit {
  let service = try! self.require_current()
  if self.active_lane_count != 0 || self.free_count == 0 {
    abort("prepared online admission phase invariant")
  }
  let prepared = match self.prepared_admission {
    Some(value) => value
    None => abort("prepared online admission owner invariant")
  }
  let lane = self.free_lanes[self.free_head]
  let old_generation = self.lane_generations[lane]
  if old_generation == 0xffffffffUL {
    service.scheduler.abort_exclusive_admission(prepared)
    self.prepared_admission = None
    return OnlineAdmissionInvalidated
  }
  let generation = old_generation + 1UL
  try
    prepared.bind_publication_route(
      try! @core.SchedulerPublicationRoute::new(lane, generation),
    )
  catch {
    _ => {
      service.scheduler.abort_exclusive_admission(prepared)
      self.prepared_admission = None
      return OnlineAdmissionInvalidated
    }
  } noraise {
    _ => ()
  }
  try
    prepared.retain_stop_tokens_into(self.lane_semantic_retentions[lane])
  catch {
    _ => {
      service.scheduler.abort_exclusive_admission(prepared)
      self.prepared_admission = None
      return OnlineAdmissionInvalidated
    }
  } noraise {
    _ => ()
  }
  if self.prepared_clock_read.read_status() == MonotonicReadFailed {
    service.scheduler.abort_exclusive_admission(prepared)
    self.release_lane_retention(lane)
    self.prepared_admission = None
    return OnlineAdmissionClockFailed
  }
  match
    service.scheduler.commit_exclusive_admission(
      prepared,
      try! self.prepared_clock_read.value(),
    ) {
    ExclusiveAdmissionExpired => {
      self.release_lane_retention(lane)
      self.prepared_admission = None
      return OnlineAdmissionExpired
    }
    ExclusiveAdmissionInvalidated => {
      self.release_lane_retention(lane)
      self.prepared_admission = None
      return OnlineAdmissionInvalidated
    }
    ExclusiveAdmissionCommitted => ()
  }
  let handle = prepared.request_handle()
  self.prepared_admission = None
  self.lane_handles[lane].push(handle)
  self.lane_generations[lane] = generation
  self.lane_request_generations[lane] = handle.request_generation()
  self.lane_cut_source_generations[lane] = handle.request_generation()
  self.lane_phases[lane] = ONLINE_LANE_LIVE
  self.free_head = (self.free_head + 1) % self.free_lanes.length()
  self.free_count -= 1
  self.active_lane_count += 1
  self.compatibility_lane = lane
  self.compatibility_generation = generation
  OnlineAdmissionCommitted
}
