///|
pub fn OnlineWorkerLease::reserve_cancel(
  self : OnlineWorkerLease,
) -> OnlineCancelReservationStart raise WorkerServiceError {
  self.compatibility_request().reserve_cancel()
}

///|
pub fn LunaOnlineWorkerRequest::reserve_cancel(
  self : LunaOnlineWorkerRequest,
) -> OnlineCancelReservationStart raise WorkerServiceError {
  let lease = self.require_live_lane()
  if lease.cancel_lane >= 0 {
    raise Invalid(Lifecycle)
  }
  let service = lease.service
  let handle = self.require_live_handle()
  let result = service.scheduler.reserve_cancel_exact(handle) catch {
    error => raise scheduler_error(error)
  }
  match result {
    CancelReserved => {
      lease.cancel_lane = self.lane
      lease.cancel_generation = self.generation
      OnlineCancelReserved
    }
    CancelNaturalTerminal => OnlineCancelNaturalTerminal
  }
}

///|
pub fn OnlineWorkerLease::take_reserved_token(
  self : OnlineWorkerLease,
) -> OnlineWorkerPublication raise WorkerServiceError {
  self.compatibility_request().take_reserved_token()
}

///|
pub fn LunaOnlineWorkerRequest::take_reserved_token(
  self : LunaOnlineWorkerRequest,
) -> OnlineWorkerPublication raise WorkerServiceError {
  let lease = self.require_live_lane()
  if lease.cancel_lane != self.lane ||
    lease.cancel_generation != self.generation {
    raise Invalid(Lifecycle)
  }
  let service = lease.service
  let handle = self.require_live_handle()
  let notice = service.scheduler.take_reserved_generated_token() catch {
    error => raise scheduler_error(error)
  }
  if notice.request_id() != handle.request_id() ||
    notice.request_generation() != lease.lane_request_generations[self.lane] {
    abort("online reserved token invariant")
  }
  if notice.position() != lease.lane_expected_positions[self.lane] {
    abort("online reserved token position invariant")
  }
  lease.lane_expected_positions[self.lane] += 1
  lease.advance_publication(notice.publication_sequence())
  OnlineWorkerPublication(
    0,
    notice.token_id(),
    notice.position(),
    notice.publication_sequence(),
    self.lane,
    self.generation,
  )
}

///|
pub fn OnlineWorkerLease::commit_cancel(
  self : OnlineWorkerLease,
) -> OnlineCancelReservationCommit raise WorkerServiceError {
  self.compatibility_request().commit_cancel()
}

///|
pub fn LunaOnlineWorkerRequest::commit_cancel(
  self : LunaOnlineWorkerRequest,
) -> OnlineCancelReservationCommit raise WorkerServiceError {
  let lease = self.require_live_lane()
  if lease.cancel_lane != self.lane ||
    lease.cancel_generation != self.generation {
    raise Invalid(Lifecycle)
  }
  let result = lease.service.scheduler.commit_cancel_reservation()
  lease.cancel_lane = -1
  lease.cancel_generation = 0UL
  match result {
    CancelCommitted(generation) => {
      let handle = self.require_live_handle()
      lease.lane_cut_source_generations[self.lane] = lease.lane_request_generations[self.lane]
      try! lease.service.scheduler.advance_terminated_handle(handle, generation)
      lease.lane_request_generations[self.lane] = generation
      lease.lane_phases[self.lane] = ONLINE_LANE_TERMINATION_PENDING
      OnlineCancelCommitted
    }
    CancelReservationAbsent => OnlineCancelReservationAbsent
  }
}

///|
pub fn OnlineWorkerLease::abort_cancel(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  self.compatibility_request().abort_cancel()
}

///|
pub fn LunaOnlineWorkerRequest::abort_cancel(
  self : LunaOnlineWorkerRequest,
) -> Unit raise WorkerServiceError {
  let lease = self.require_live_lane()
  if lease.cancel_lane != self.lane ||
    lease.cancel_generation != self.generation {
    raise Invalid(Lifecycle)
  }
  lease.service.scheduler.abort_cancel_reservation()
  lease.cancel_lane = -1
  lease.cancel_generation = 0UL
}

///|
pub fn OnlineWorkerLease::expire(
  self : OnlineWorkerLease,
) -> OnlineExpiry raise WorkerServiceError {
  self.compatibility_request().expire()
}

///|
pub fn LunaOnlineWorkerRequest::expire(
  self : LunaOnlineWorkerRequest,
) -> OnlineExpiry raise WorkerServiceError {
  let lease = self.require_live_lane()
  let service = lease.service
  let handle = self.require_live_handle()
  let now = match
    read_expiry_clock(
      service.scheduler.has_exact_natural_terminal(handle),
      lease,
      read_online_lease_clock,
    ) {
    ExpiryNaturalTerminal => return OnlineDeadlineNaturalTerminal
    ExpiryClockSample(value) => value
  }
  let expired = service.scheduler.expire_exact_or_natural(handle, now) catch {
    error => raise scheduler_error(error)
  }
  match expired {
    ExactDeadlinePending => OnlineDeadlinePending
    ExactDeadlineNaturalTerminal => OnlineDeadlineNaturalTerminal
    ExactDeadlineExpired(generation) => {
      lease.lane_cut_source_generations[self.lane] = lease.lane_request_generations[self.lane]
      try! service.scheduler.advance_terminated_handle(handle, generation)
      lease.lane_request_generations[self.lane] = generation
      lease.lane_phases[self.lane] = ONLINE_LANE_TERMINATION_PENDING
      OnlineDeadlineCommitted
    }
  }
}

///|
fn[T] read_expiry_clock(
  has_natural_terminal : Bool,
  owner : T,
  read_clock : (T) -> UInt64 raise WorkerServiceError,
) -> OnlineExpiryClockPreflight raise WorkerServiceError {
  if has_natural_terminal {
    ExpiryNaturalTerminal
  } else {
    ExpiryClockSample(read_clock(owner))
  }
}

///|
fn read_online_lease_clock(
  lease : OnlineWorkerLease,
) -> UInt64 raise WorkerServiceError {
  lease.clock.now_millis() catch {
    _ => raise Invalid(Clock)
  }
}
