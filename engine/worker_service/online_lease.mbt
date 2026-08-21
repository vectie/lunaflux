///|
fn WorkerService::require_online_epoch(
  self : WorkerService,
  epoch : UInt64,
) -> Unit raise WorkerServiceError {
  if self.ownership_mode != OnlineActiveOwnership ||
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
  if self.request_phase != OnlineRequestLive {
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
  if self.request_phase != OnlineRequestLive &&
    self.request_phase != OnlineRequestTerminationPending {
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
  self.ownership_mode == OnlineEligibleOwnership &&
  self.lifecycle == ServiceReady &&
  self.raw_outstanding_count() == 0 &&
  self.scheduler.is_pristine() &&
  self.require_process().has_exchange_capacity()
}

///|
pub fn OnlineWorkerLease::progress(
  self : OnlineWorkerLease,
) -> WorkerServiceProgress raise WorkerServiceError {
  let service = self.require_current()
  if self.request_phase == OnlineRequestLive {
    if service.scheduler.generated_token_count() != 0 ||
      service.scheduler.terminal_notice_count() != 0 {
      raise Invalid(Lifecycle)
    }
    return service.progress_impl()
  }
  if (
      self.request_phase == OnlineRequestTerminationPending ||
      self.request_phase == OnlineRequestTerminal
    ) &&
    service.raw_outstanding_count() != 0 {
    return service.progress_existing_flight_impl()
  }
  raise Invalid(Lifecycle)
}

///|
fn classify_online_worker_error(error : WorkerServiceError) -> Int {
  match error {
    Scheduler(Invalid(issue=Backpressured, ..)) => 0
    RootBoundProcess(RootBoundFailure(Process(Backpressured))) => 0
    RootBoundProcess(
      RootBoundFailure(
        Process(Wire(_) | Protocol(_) | Invalid(Completion | Sequence))
      )
    ) => 1
    RootBoundProcess(_) => 2
    Scheduler(WorkerProtocol(_)) | Invalid(WorkerOutput) => 1
    Scheduler(
      Invalid(
        field=Completion
        | RequestState,
        issue=Mismatch
        | OutOfOrder
        | InvalidTransition
        | Stale
      )
    ) => 1
    Invalid(Sequence) => 1
    _ => 2
  }
}

///|
/// Advances one online worker transition and returns only the product-safe
/// disposition needed by the owning aggregate. A physical process failure is
/// classified only after the lower service has retained recovery authority;
/// a received completion validation failure is classified as invalid output.
pub fn OnlineWorkerLease::progress_status(
  self : OnlineWorkerLease,
) -> OnlineWorkerStep raise WorkerServiceError {
  let service = self.require_current()
  try self.progress() catch {
    error => {
      let disposition = classify_online_worker_error(error)
      if disposition == 0 {
        return OnlineWorkerRetryable
      }
      if service.lifecycle == ServiceReady {
        // Retain the exact lower owner but never close/reap from the reactor
        // step. The explicit terminalization API performs that work.
        service.lifecycle = ServiceRecovering
      }
      if disposition == 1 {
        OnlineWorkerOutputInvalid
      } else {
        OnlineWorkerUnavailable
      }
    }
  } noraise {
    _ => OnlineWorkerAdvanced
  }
}

///|
pub fn OnlineWorkerLease::has_publication(
  self : OnlineWorkerLease,
) -> Bool raise WorkerServiceError {
  let service = self.require_current()
  if self.request_phase == OnlineRequestTerminationPending &&
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
  if self.request_phase == OnlineRequestTerminationPending &&
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
        if self.request_phase != OnlineRequestTerminationPending ||
          publication.request_generation() != self.cut_source_generation {
          abort("online token generation invariant")
        }
      }
      let notice = try! publication.generated_token_notice()
      if notice.position() != self.expected_position {
        abort("online token position invariant")
      }
      self.expected_position += 1
      let kind = if self.request_phase == OnlineRequestLive { 0 } else { 2 }
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
      self.request_phase = OnlineRequestTerminal
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
      self.request_phase = OnlineRequestTerminationPending
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
      self.request_phase = OnlineRequestTerminationPending
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
