///|
const ONLINE_LANE_EMPTY : Int = 0

///|
const ONLINE_LANE_LIVE : Int = 1

///|
const ONLINE_LANE_TERMINATION_PENDING : Int = 2

///|
const ONLINE_LANE_TERMINAL : Int = 3

///|
fn luna_online_worker_admission(
  kind : LunaOnlineWorkerAdmissionKind,
  lease : OnlineWorkerLease,
  lane : Int,
  generation : UInt64,
) -> LunaOnlineWorkerAdmission {
  {
    disposition: match kind {
      LunaOnlineWorkerAdmitted => 0
      LunaOnlineWorkerSaturated => 1
      LunaOnlineWorkerDraining => 2
    },
    lease,
    lease_epoch: lease.epoch,
    lane,
    generation,
  }
}

///|
pub fn LunaOnlineWorkerAdmission::kind(
  self : LunaOnlineWorkerAdmission,
) -> LunaOnlineWorkerAdmissionKind {
  match self.disposition {
    0 => LunaOnlineWorkerAdmitted
    1 => LunaOnlineWorkerSaturated
    2 => LunaOnlineWorkerDraining
    _ => abort("Luna online worker admission invariant")
  }
}

///|
pub fn LunaOnlineWorkerAdmission::request(
  self : LunaOnlineWorkerAdmission,
) -> LunaOnlineWorkerRequest raise WorkerServiceError {
  if self.disposition != 0 || self.lane < 0 || self.generation == 0UL {
    raise Invalid(Lifecycle)
  }
  let request : LunaOnlineWorkerRequest = {
    lease: self.lease,
    lease_epoch: self.lease_epoch,
    lane: self.lane,
    generation: self.generation,
  }
  ignore(request.require_live_lane())
  request
}

///|
fn LunaOnlineWorkerRequest::require_lane(
  self : LunaOnlineWorkerRequest,
) -> OnlineWorkerLease raise WorkerServiceError {
  let lease = self.lease
  lease.service.require_online_epoch(self.lease_epoch)
  if self.lane < 0 ||
    self.lane >= lease.lane_phases.length() ||
    self.generation == 0UL ||
    lease.lane_generations[self.lane] != self.generation ||
    lease.lane_phases[self.lane] == ONLINE_LANE_EMPTY {
    raise Invalid(Lifecycle)
  }
  lease
}

///|
fn LunaOnlineWorkerRequest::require_live_lane(
  self : LunaOnlineWorkerRequest,
) -> OnlineWorkerLease raise WorkerServiceError {
  let lease = self.require_lane()
  if lease.lane_phases[self.lane] != ONLINE_LANE_LIVE {
    raise Invalid(Lifecycle)
  }
  lease
}

///|
pub fn LunaOnlineWorkerRequest::is_live(self : LunaOnlineWorkerRequest) -> Bool {
  let lease = self.lease
  self.lease_epoch != 0UL &&
  lease.service.ownership_mode == OnlineActiveOwnership &&
  lease.service.online_epoch == self.lease_epoch &&
  self.lane >= 0 &&
  self.lane < lease.lane_phases.length() &&
  self.generation != 0UL &&
  lease.lane_generations[self.lane] == self.generation &&
  lease.lane_phases[self.lane] != ONLINE_LANE_EMPTY
}

///|
/// Authenticates whether this exact request owns an opaque publication route.
/// It exposes neither the worker lane nor either generation scalar.
pub fn LunaOnlineWorkerRequest::matches_publication_route(
  self : LunaOnlineWorkerRequest,
  route : @core.SchedulerPublicationRoute,
) -> Bool {
  self.is_live() &&
  route.lane() == self.lane &&
  route.generation() == self.generation
}

///|
/// Returns this exact retained request's opaque publication route. The scalar
/// value grants no scheduler, worker-lane, or request mutation authority.
pub fn LunaOnlineWorkerRequest::publication_route(
  self : LunaOnlineWorkerRequest,
) -> @core.SchedulerPublicationRoute raise WorkerServiceError {
  ignore(self.require_lane())
  @core.SchedulerPublicationRoute::new(self.lane, self.generation) catch {
    error => raise scheduler_error(error)
  }
}

///|
fn OnlineWorkerLease::release_lane_retention(
  self : OnlineWorkerLease,
  lane : Int,
) -> Unit {
  if self.lane_semantic_retentions[lane].is_live() {
    try! self.lane_semantic_retentions[lane].release()
  }
}

///|
fn OnlineWorkerLease::release_lane(
  self : OnlineWorkerLease,
  lane : Int,
) -> Unit {
  self.release_lane_retention(lane)
  self.lane_handles[lane].clear()
  self.lane_request_generations[lane] = @inference.RequestGeneration::initial()
  self.lane_cut_source_generations[lane] = @inference.RequestGeneration::initial()
  self.lane_restart_dispositions[lane] = 0
  self.lane_expected_positions[lane] = 0
  self.lane_phases[lane] = ONLINE_LANE_EMPTY
  let tail = (self.free_head + self.free_count) % self.free_lanes.length()
  self.free_lanes[tail] = lane
  self.free_count += 1
  self.active_lane_count -= 1
  if self.compatibility_lane == lane {
    self.compatibility_lane = -1
    self.compatibility_generation = 0UL
  }
}

///|
/// Attempts one request admission into startup-preallocated worker storage.
/// Saturation and draining consume neither the request nor a fixed lane.
pub fn OnlineWorkerLease::try_admit_request(
  self : OnlineWorkerLease,
  request : @core.TokenizedRequest,
) -> LunaOnlineWorkerAdmission raise WorkerServiceError {
  let service = self.require_current()
  if service.lifecycle != ServiceReady {
    return luna_online_worker_admission(LunaOnlineWorkerDraining, self, -1, 0UL)
  }
  if self.prepared_admission is Some(_) || self.cancel_lane >= 0 {
    raise Invalid(Lifecycle)
  }
  if self.free_count == 0 {
    return luna_online_worker_admission(
      LunaOnlineWorkerSaturated,
      self,
      -1,
      0UL,
    )
  }
  let lane = self.free_lanes[self.free_head]
  let old_generation = self.lane_generations[lane]
  if old_generation == 0xffffffffUL {
    return luna_online_worker_admission(LunaOnlineWorkerDraining, self, -1, 0UL)
  }
  let generation = old_generation + 1UL
  let prepared = service.scheduler.prepare_admission(request)
  let route = @core.SchedulerPublicationRoute::new(lane, generation) catch {
    error => raise scheduler_error(error)
  }
  prepared.bind_publication_route(route) catch {
    error => raise scheduler_error(error)
  }
  prepared.retain_stop_tokens_into(self.lane_semantic_retentions[lane]) catch {
    error => raise scheduler_error(error)
  }
  let now = self.clock.now_millis() catch {
    _ => {
      self.release_lane_retention(lane)
      raise Invalid(Clock)
    }
  }
  service.scheduler.commit_admission(prepared, now) catch {
    error => {
      self.release_lane_retention(lane)
      raise scheduler_error(error)
    }
  }
  let handle = prepared.request_handle() catch {
    _ => abort("Luna online worker committed handle invariant")
  }
  self.lane_handles[lane].push(handle)
  self.lane_generations[lane] = generation
  self.lane_request_generations[lane] = handle.request_generation()
  self.lane_cut_source_generations[lane] = handle.request_generation()
  self.lane_restart_dispositions[lane] = 0
  self.lane_expected_positions[lane] = 0
  self.lane_phases[lane] = ONLINE_LANE_LIVE
  self.free_head = (self.free_head + 1) % self.free_lanes.length()
  self.free_count -= 1
  self.active_lane_count += 1
  luna_online_worker_admission(LunaOnlineWorkerAdmitted, self, lane, generation)
}

///|
pub fn OnlineWorkerLease::request_capacity(self : OnlineWorkerLease) -> Int {
  self.lane_phases.length()
}

///|
pub fn OnlineWorkerLease::active_request_count(
  self : OnlineWorkerLease,
) -> Int raise WorkerServiceError {
  ignore(self.require_current())
  self.active_lane_count
}

///|
pub fn LunaOnlineWorkerRequest::retire_terminal(
  self : LunaOnlineWorkerRequest,
) -> Unit raise WorkerServiceError {
  let lease = self.require_lane()
  if lease.lane_phases[self.lane] != ONLINE_LANE_TERMINAL ||
    lease.cancel_lane >= 0 ||
    lease.service.lifecycle != ServiceReady {
    raise Invalid(Lifecycle)
  }
  lease.release_lane(self.lane)
}
