///|
fn WorkerService::require_online_epoch(
  self : WorkerService,
  epoch : UInt64,
) -> Unit raise WorkerServiceError {
  if self.ownership_mode != SERVICE_MODE_ONLINE_ACTIVE ||
    epoch == 0UL ||
    self.online_epoch != epoch {
    raise Invalid(OwnershipMode)
  }
}

///|
fn OnlineWorkerLease::require_current(
  self : OnlineWorkerLease,
) -> WorkerService raise WorkerServiceError {
  self.service.require_online_epoch(self.epoch)
  self.service
}

///|
fn OnlineWorkerLease::require_live_handle(
  self : OnlineWorkerLease,
) -> @core.RequestHandle raise WorkerServiceError {
  ignore(self.require_current())
  if self.request_phase != ONLINE_REQUEST_LIVE {
    raise Invalid(Lifecycle)
  }
  match self.handle {
    Some(handle) => handle
    None => abort("online worker handle invariant")
  }
}

///|
fn OnlineWorkerLease::require_pending_handle(
  self : OnlineWorkerLease,
) -> @core.RequestHandle raise WorkerServiceError {
  ignore(self.require_current())
  if self.request_phase != ONLINE_REQUEST_LIVE &&
    self.request_phase != ONLINE_REQUEST_TERMINATION_PENDING {
    raise Invalid(Lifecycle)
  }
  match self.handle {
    Some(handle) => handle
    None => abort("online worker handle invariant")
  }
}

///|
fn WorkerService::can_claim_raw(self : WorkerService) -> Bool {
  self.production_owned &&
  self.ownership_mode == SERVICE_MODE_ONLINE_ELIGIBLE &&
  self.lifecycle == ServiceReady &&
  self.raw_outstanding_count() == 0 &&
  self.scheduler.is_pristine() &&
  self.require_process().has_exchange_capacity()
}

///|
/// Samples the retained monotonic clock immediately before the scheduler
/// mutation and binds the exact returned handle privately.
pub fn OnlineWorkerLease::admit(
  self : OnlineWorkerLease,
  request : @core.TokenizedRequest,
) -> Unit raise WorkerServiceError {
  let service = self.require_current()
  if self.request_phase != ONLINE_REQUEST_EMPTY {
    raise Invalid(Lifecycle)
  }
  let prepared = service.scheduler.prepare_admission(request)
  let now = self.clock.now_millis() catch { _ => raise Invalid(Clock) }
  service.scheduler.commit_admission(prepared, now) catch {
    error => raise scheduler_error(error)
  }
  let handle = try! prepared.request_handle()
  self.handle = Some(handle)
  self.request_generation = handle.request_generation()
  self.request_phase = ONLINE_REQUEST_LIVE
}

///|
pub fn OnlineWorkerLease::progress(
  self : OnlineWorkerLease,
) -> WorkerServiceProgress raise WorkerServiceError {
  let service = self.require_current()
  if self.request_phase == ONLINE_REQUEST_LIVE {
    if service.scheduler.generated_token_count() != 0 ||
      service.scheduler.terminal_notice_count() != 0 {
      raise Invalid(Lifecycle)
    }
    return service.progress_impl()
  }
  if (
      self.request_phase == ONLINE_REQUEST_TERMINATION_PENDING ||
      self.request_phase == ONLINE_REQUEST_TERMINAL
    ) &&
    service.raw_outstanding_count() != 0 {
    return service.progress_existing_flight_impl()
  }
  raise Invalid(Lifecycle)
}

///|
pub fn OnlineWorkerLease::has_publication(
  self : OnlineWorkerLease,
) -> Bool raise WorkerServiceError {
  let service = self.require_current()
  if self.request_phase == ONLINE_REQUEST_TERMINATION_PENDING &&
    service.raw_outstanding_count() != 0 {
    return false
  }
  service.scheduler.generated_token_count() > 0 ||
  service.scheduler.terminal_notice_count() > 0
}

///|
pub fn OnlineWorkerPublication::kind(
  self : OnlineWorkerPublication,
) -> OnlineWorkerPublicationKind {
  if self.0 == 0 {
    OnlineGeneratedTokenPublication
  } else if self.0 == 2 {
    OnlineSuppressedTokenPublication
  } else {
    OnlineTerminalPublication
  }
}

///|
pub fn OnlineWorkerPublication::token_id(
  self : OnlineWorkerPublication,
) -> Int raise WorkerServiceError {
  if self.0 != 0 {
    raise Invalid(Sequence)
  }
  self.1
}

///|
pub fn OnlineWorkerPublication::position(
  self : OnlineWorkerPublication,
) -> Int raise WorkerServiceError {
  if self.0 != 0 {
    raise Invalid(Sequence)
  }
  self.2
}

///|
pub fn OnlineWorkerPublication::terminal_reason(
  self : OnlineWorkerPublication,
) -> OnlineWorkerTerminalReason raise WorkerServiceError {
  if self.0 != 1 {
    raise Invalid(Sequence)
  }
  match self.2 {
    0 => OnlineCancelledByCaller
    1 => OnlineDeadlineReached
    2 => OnlineStopTokenReached
    3 => OnlineMaximumOutputReached
    4 => OnlineWorkerFailed
    _ => abort("online terminal reason invariant")
  }
}

///|
pub fn OnlineWorkerPublication::publication_sequence(
  self : OnlineWorkerPublication,
) -> UInt64 {
  self.3
}

///|
fn online_terminal_code(reason : @core.TerminalReason) -> Int {
  match reason {
    CancelledByCaller => 0
    DeadlineReached => 1
    StopTokenReached => 2
    MaximumOutputReached => 3
    WorkerFailed => 4
  }
}

///|
fn OnlineWorkerLease::advance_publication(
  self : OnlineWorkerLease,
  sequence : UInt64,
) -> Unit {
  if sequence == 0UL || sequence != self.publication_cursor + 1UL {
    abort("online publication sequence invariant")
  }
  self.publication_cursor = sequence
  self.service.online_publication_cursor = sequence
}

///|
pub fn OnlineWorkerLease::take_publication(
  self : OnlineWorkerLease,
) -> OnlineWorkerPublication raise WorkerServiceError {
  let service = self.require_current()
  let handle = self.require_pending_handle()
  if self.request_phase == ONLINE_REQUEST_TERMINATION_PENDING &&
    service.raw_outstanding_count() != 0 {
    raise Invalid(Lifecycle)
  }
  let publication = service.scheduler.take_next_publication() catch {
    error => raise scheduler_error(error)
  }
  if publication.request_id() != handle.request_id() {
    abort("online publication request invariant")
  }
  self.advance_publication(publication.publication_sequence())
  match publication.kind() {
    GeneratedTokenPublication => {
      if publication.request_generation() != self.request_generation {
        if self.request_phase != ONLINE_REQUEST_TERMINATION_PENDING ||
          publication.request_generation() != self.cut_source_generation {
          abort("online token generation invariant")
        }
      }
      let notice = try! publication.generated_token_notice()
      if notice.position() != self.expected_position {
        abort("online token position invariant")
      }
      self.expected_position += 1
      let kind = if self.request_phase == ONLINE_REQUEST_LIVE { 0 } else { 2 }
      OnlineWorkerPublication(
        kind,
        notice.token_id(),
        notice.position(),
        notice.publication_sequence(),
      )
    }
    TerminalPublication => {
      let notice = try! publication.terminal_notice()
      if notice.request_generation() != self.request_generation {
        abort("online terminal generation invariant")
      }
      self.request_phase = ONLINE_REQUEST_TERMINAL
      OnlineWorkerPublication(
        1,
        0,
        online_terminal_code(notice.reason()),
        notice.publication_sequence(),
      )
    }
  }
}

///|
pub fn OnlineWorkerLease::reserve_cancel(
  self : OnlineWorkerLease,
) -> OnlineCancelReservationStart raise WorkerServiceError {
  let service = self.require_current()
  let handle = self.require_live_handle()
  let result = service.scheduler.reserve_cancel_exact(handle) catch {
    error => raise scheduler_error(error)
  }
  match result {
    CancelReserved => {
      self.cancel_reserved = true
      OnlineCancelReserved
    }
    CancelNaturalTerminal => OnlineCancelNaturalTerminal
  }
}

///|
pub fn OnlineWorkerLease::take_reserved_token(
  self : OnlineWorkerLease,
) -> OnlineWorkerPublication raise WorkerServiceError {
  let service = self.require_current()
  if !self.cancel_reserved {
    raise Invalid(Lifecycle)
  }
  let handle = self.require_live_handle()
  let notice = service.scheduler.take_reserved_generated_token() catch {
    error => raise scheduler_error(error)
  }
  if notice.request_id() != handle.request_id() ||
    notice.request_generation() != self.request_generation {
    abort("online reserved token invariant")
  }
  if notice.position() != self.expected_position {
    abort("online reserved token position invariant")
  }
  self.expected_position += 1
  self.advance_publication(notice.publication_sequence())
  OnlineWorkerPublication(
    0,
    notice.token_id(),
    notice.position(),
    notice.publication_sequence(),
  )
}

///|
pub fn OnlineWorkerLease::commit_cancel(
  self : OnlineWorkerLease,
) -> OnlineCancelReservationCommit raise WorkerServiceError {
  ignore(self.require_current())
  let result = self.service.scheduler.commit_cancel_reservation()
  self.cancel_reserved = false
  match result {
    CancelCommitted(generation) => {
      let handle = match self.handle {
        Some(value) => value
        None => abort("online cancel handle invariant")
      }
      self.cut_source_generation = self.request_generation
      try! self.service.scheduler.advance_terminated_handle(handle, generation)
      self.request_generation = generation
      self.request_phase = ONLINE_REQUEST_TERMINATION_PENDING
      OnlineCancelCommitted
    }
    CancelReservationAbsent => OnlineCancelReservationAbsent
  }
}

///|
pub fn OnlineWorkerLease::abort_cancel(
  self : OnlineWorkerLease,
) -> Unit raise WorkerServiceError {
  ignore(self.require_current())
  self.service.scheduler.abort_cancel_reservation()
  self.cancel_reserved = false
}

///|
pub fn OnlineWorkerLease::expire(
  self : OnlineWorkerLease,
) -> OnlineExpiry raise WorkerServiceError {
  let service = self.require_current()
  let handle = self.require_live_handle()
  let now = match
    read_expiry_clock(
      service.scheduler.has_exact_natural_terminal(handle),
      self,
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
      self.cut_source_generation = self.request_generation
      try! service.scheduler.advance_terminated_handle(handle, generation)
      self.request_generation = generation
      self.request_phase = ONLINE_REQUEST_TERMINATION_PENDING
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
