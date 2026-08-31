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
fn LunaOnlineWorkerRequest::require_live_handle(
  self : LunaOnlineWorkerRequest,
) -> @core.RequestHandle raise WorkerServiceError {
  let lease = self.require_live_lane()
  if lease.lane_handles[self.lane].length() != 1 {
    abort("Luna online worker handle invariant")
  }
  lease.lane_handles[self.lane][0]
}

///|
fn LunaOnlineWorkerRequest::require_pending_handle(
  self : LunaOnlineWorkerRequest,
) -> @core.RequestHandle raise WorkerServiceError {
  let lease = self.require_lane()
  if lease.lane_phases[self.lane] != ONLINE_LANE_LIVE &&
    lease.lane_phases[self.lane] != ONLINE_LANE_TERMINATION_PENDING {
    raise Invalid(Lifecycle)
  }
  if lease.lane_handles[self.lane].length() != 1 {
    abort("Luna online worker handle invariant")
  }
  lease.lane_handles[self.lane][0]
}

///|
fn OnlineWorkerLease::compatibility_request(
  self : OnlineWorkerLease,
) -> LunaOnlineWorkerRequest raise WorkerServiceError {
  ignore(self.require_current())
  if self.active_lane_count != 1 ||
    self.compatibility_lane < 0 ||
    self.compatibility_generation == 0UL {
    raise Invalid(Lifecycle)
  }
  let request : LunaOnlineWorkerRequest = {
    lease: self,
    lease_epoch: self.epoch,
    lane: self.compatibility_lane,
    generation: self.compatibility_generation,
  }
  ignore(request.require_lane())
  request
}

///|
fn WorkerService::can_claim_raw(self : WorkerService) -> Bool {
  self.production_owned &&
  self.ownership_mode == OnlineEligibleOwnership &&
  self.lifecycle == ServiceReady &&
  self.raw_outstanding_count() == 0 &&
  self.scheduler.is_pristine() &&
  self.require_transport().has_exchange_capacity()
}

///|
pub fn OnlineWorkerLease::progress(
  self : OnlineWorkerLease,
) -> WorkerServiceProgress raise WorkerServiceError {
  let service = self.require_current()
  if self.active_lane_count == 0 {
    raise Invalid(Lifecycle)
  }
  // A logical cut may publish its terminal while the exact physical exchange
  // is still live. Retire that already-owned flight before applying the
  // publication backpressure that prevents construction of a new plan.
  if service.raw_outstanding_count() != 0 {
    return service.progress_existing_flight_impl()
  }
  if service.scheduler.generated_token_count() != 0 ||
    service.scheduler.terminal_notice_count() != 0 {
    raise Invalid(Lifecycle)
  }
  if service.scheduler.live_request_count() == 0 {
    raise Invalid(Lifecycle)
  }
  let progress = service.progress_impl()
  match progress {
    // This commit is emitted only by the currently owned physical flight.
    // The private replacement generation additionally requires a sequence
    // newer than the scheduler predecessor captured after that replacement
    // spawned, so a stale completion cannot reset crash-loop attempts.
    ServiceCommitted(sequence) =>
      self.record_stable_replacement_commit(service, sequence)
    _ => ()
  }
  progress
}

///|
fn classify_online_worker_error(error : WorkerServiceError) -> Int {
  match error {
    Scheduler(Invalid(issue=Backpressured, ..)) => 0
    PhysicalTransport(PhysicalBackpressured) => 0
    PhysicalTransport(PhysicalInvalidOutput) => 1
    PhysicalTransport(_) => 2
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
    progress =>
      match progress {
        ServiceIdle | ServiceBackpressured | ServicePending(_) =>
          OnlineWorkerWaiting
        _ => OnlineWorkerAdvanced
      }
  }
}

///|
pub fn OnlineWorkerLease::has_publication(
  self : OnlineWorkerLease,
) -> Bool raise WorkerServiceError {
  let service = self.require_current()
  service.raw_outstanding_count() == 0 &&
  (
    service.scheduler.generated_token_count() > 0 ||
    service.scheduler.terminal_notice_count() > 0
  )
}

///|
/// Returns the authenticated fixed-lane route of the globally oldest
/// publication without consuming it.
pub fn OnlineWorkerLease::next_publication_route(
  self : OnlineWorkerLease,
) -> @core.SchedulerPublicationRoute raise WorkerServiceError {
  let service = self.require_current()
  if service.raw_outstanding_count() != 0 {
    raise Invalid(Lifecycle)
  }
  service.scheduler.next_publication_route() catch {
    error => raise scheduler_error(error)
  }
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
/// Returns the exact request-local full-page prefix reuse captured by the
/// scheduler before terminal slot retirement. Token and suppressed-token
/// publications do not own terminal usage and reject this accessor.
pub fn OnlineWorkerPublication::cached_input_tokens(
  self : OnlineWorkerPublication,
) -> Int raise WorkerServiceError {
  if self.0 != 1 || self.1 < 0 {
    raise Invalid(Sequence)
  }
  self.1
}

///|
pub fn OnlineWorkerPublication::publication_sequence(
  self : OnlineWorkerPublication,
) -> UInt64 {
  self.3
}

///|
/// Returns the opaque caller route copied before scheduler admission. It is
/// sufficient for an owning aggregate to select one fixed request lane and
/// grants no scheduler or request mutation authority.
pub fn OnlineWorkerPublication::route(
  self : OnlineWorkerPublication,
) -> @core.SchedulerPublicationRoute {
  try! @core.SchedulerPublicationRoute::new(self.4, self.5)
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
  if service.raw_outstanding_count() != 0 {
    raise Invalid(Lifecycle)
  }
  let publication = service.scheduler.take_next_publication() catch {
    error => raise scheduler_error(error)
  }
  let route = publication.route()
  let lane = route.lane()
  let generation = route.generation()
  if lane < 0 ||
    lane >= self.lane_phases.length() ||
    generation == 0UL ||
    self.lane_generations[lane] != generation ||
    self.lane_phases[lane] == ONLINE_LANE_EMPTY ||
    self.lane_handles[lane].length() != 1 {
    abort("Luna online publication route invariant")
  }
  let handle = self.lane_handles[lane][0]
  if publication.request_id() != handle.request_id() {
    abort("online publication request invariant")
  }
  self.advance_publication(publication.publication_sequence())
  match publication.kind() {
    GeneratedTokenPublication => {
      if publication.request_generation() != self.lane_request_generations[lane] {
        if self.lane_phases[lane] != ONLINE_LANE_TERMINATION_PENDING ||
          publication.request_generation() !=
          self.lane_cut_source_generations[lane] {
          abort("online token generation invariant")
        }
      }
      let notice = try! publication.generated_token_notice()
      if notice.position() != self.lane_expected_positions[lane] {
        abort("online token position invariant")
      }
      self.lane_expected_positions[lane] += 1
      let kind = if self.lane_phases[lane] == ONLINE_LANE_LIVE { 0 } else { 2 }
      OnlineWorkerPublication(
        kind,
        notice.token_id(),
        notice.position(),
        notice.publication_sequence(),
        lane,
        generation,
      )
    }
    TerminalPublication => {
      let notice = try! publication.terminal_notice()
      if notice.request_generation() != self.lane_request_generations[lane] {
        if self.lane_phases[lane] != ONLINE_LANE_TERMINATION_PENDING ||
          notice.reason() != WorkerFailed {
          abort("online terminal generation invariant")
        }
        self.lane_request_generations[lane] = notice.request_generation()
      }
      self.lane_phases[lane] = ONLINE_LANE_TERMINAL
      self.lane_restart_dispositions[lane] = 0
      OnlineWorkerPublication(
        1,
        notice.cached_input_tokens(),
        online_terminal_code(notice.reason()),
        notice.publication_sequence(),
        lane,
        generation,
      )
    }
  }
}
