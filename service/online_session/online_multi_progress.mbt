///|
fn LunaOnlineInstance::multi_begin_cancel(
  self : LunaOnlineInstance,
  lane_index : Int,
  cut : SessionTerminalCut,
) -> Bool {
  let lane = self.multi.lanes[lane_index]
  let request = self.multi_worker_request(lane_index)
  let reservation = try! request.reserve_cancel()
  match reservation {
    OnlineCancelNaturalTerminal => false
    OnlineCancelReserved => {
      let committed = try! request.commit_cancel()
      match committed {
        OnlineCancelCommitted => lane.terminal_cut = cut
        OnlineCancelReservationAbsent =>
          abort("Luna online multi cancellation invariant")
      }
      true
    }
  }
}

///|
fn LunaOnlineInstance::multi_publish_usage(
  self : LunaOnlineInstance,
  lane_index : Int,
  kind : SessionEventKind,
) -> Unit {
  let lane = self.multi.lanes[lane_index]
  try! self.events.publish_usage(
    input_tokens=lane.input_tokens,
    cached_input_tokens=lane.cached_input_tokens,
    output_tokens=lane.output_tokens,
  )
  self.mark_multi_event_published(lane_index, kind)
}

///|
fn LunaOnlineInstance::multi_prepare_cut_terminal(
  self : LunaOnlineInstance,
  lane_index : Int,
) -> Unit {
  let lane = self.multi.lanes[lane_index]
  let kind = if lane.terminal_cut == CallerCancellationCut {
    lane.finish_kind = FinishCancelled
    CompletionUsageEvent
  } else {
    if lane.failure_kind == NoLatchedFailure {
      lane.failure_kind = match lane.terminal_cut {
        DeadlineFailureCut => DeadlineLatchedFailure
        WorkerFailureCut => WorkerLatchedFailure
        OutputCancellationCut | NaturalOutputFailureCut => OutputLatchedFailure
        NoTerminalCut | CallerCancellationCut =>
          abort("Luna online multi terminal invariant")
      }
    }
    FailureUsageEvent
  }
  lane.terminal_cut = NoTerminalCut
  lane.restarted_failure_terminal_pending = false
  self.multi_publish_usage(lane_index, kind)
}

///|
fn LunaOnlineInstance::multi_prepare_natural_terminal(
  self : LunaOnlineInstance,
  lane_index : Int,
  reason : @worker_service.OnlineWorkerTerminalReason,
) -> Unit raise LunaOnlineInstanceError {
  let lane = self.multi.lanes[lane_index]
  if reason == OnlineWorkerFailed {
    lane.failure_kind = WorkerLatchedFailure
    lane.final_tail_length = 0
    lane.recovery_close = false
    self.multi_publish_usage(lane_index, FailureUsageEvent)
    return
  }
  lane.finish_kind = match reason {
    OnlineStopTokenReached => FinishStopSequence
    OnlineMaximumOutputReached => FinishTokenLimit
    _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
  }
  let tail = self
    .multi_claim_owner(lane_index)
    .finish_into_status(lane.output, destination_offset=0)
  if !tail.is_ready() {
    lane.failure_kind = OutputLatchedFailure
    lane.final_tail_length = 0
    self.multi_publish_usage(lane_index, FailureUsageEvent)
    return
  }
  lane.final_tail_length = tail.copied_bytes()
  self.multi_publish_usage(lane_index, CompletionUsageEvent)
}

///|
fn LunaOnlineInstance::multi_prepare_worker_failure_race_terminal(
  self : LunaOnlineInstance,
  lane_index : Int,
  reason : @worker_service.OnlineWorkerTerminalReason,
) -> Unit raise LunaOnlineInstanceError {
  let lane = self.multi.lanes[lane_index]
  lane.terminal_cut = NoTerminalCut
  lane.restarted_failure_terminal_pending = false
  lane.recovery_close = false
  match reason {
    OnlineCancelledByCaller => {
      lane.failure_kind = NoLatchedFailure
      lane.finish_kind = FinishCancelled
      lane.final_tail_length = 0
      self.multi_publish_usage(lane_index, CompletionUsageEvent)
    }
    OnlineDeadlineReached => {
      lane.failure_kind = DeadlineLatchedFailure
      lane.final_tail_length = 0
      self.multi_publish_usage(lane_index, FailureUsageEvent)
    }
    OnlineStopTokenReached | OnlineMaximumOutputReached => {
      lane.failure_kind = NoLatchedFailure
      self.multi_prepare_natural_terminal(lane_index, reason)
    }
    OnlineWorkerFailed => abort("Luna online multi worker race invariant")
  }
}

///|
fn LunaOnlineInstance::multi_prepare_string_terminal(
  self : LunaOnlineInstance,
  lane_index : Int,
) -> Unit {
  let lane = self.multi.lanes[lane_index]
  lane.string_cut = NoStringCut
  lane.finish_kind = FinishStopSequence
  lane.final_tail_length = 0
  self.multi_publish_usage(lane_index, CompletionUsageEvent)
}

///|
fn LunaOnlineInstance::multi_begin_output_failure(
  self : LunaOnlineInstance,
  lane_index : Int,
) -> Unit {
  let lane = self.multi.lanes[lane_index]
  lane.failure_kind = OutputLatchedFailure
  if !self.multi_begin_cancel(lane_index, OutputCancellationCut) {
    lane.terminal_cut = NaturalOutputFailureCut
  }
}

///|
fn LunaOnlineInstance::multi_decode_token(
  self : LunaOnlineInstance,
  lane_index : Int,
  publication : @worker_service.OnlineWorkerPublication,
) -> Unit {
  let lane = self.multi.lanes[lane_index]
  let claim = self.multi_claim_owner(lane_index)
  let token_id = try! publication.token_id()
  lane.output_tokens += 1
  if claim.is_stop_token(token_id) {
    lane.string_cut = NaturalStringCut
    return
  }
  let step = claim.push_token_into_status(
    token_id,
    lane.output,
    destination_offset=0,
  )
  if !step.is_ready() {
    self.multi_begin_output_failure(lane_index)
    return
  }
  if step.stop_matched() {
    if self.multi_begin_cancel(lane_index, OutputCancellationCut) {
      lane.string_cut = CancelledStringCut
      lane.terminal_cut = NoTerminalCut
    } else {
      lane.string_cut = NaturalStringCut
    }
    if step.copied_bytes() == 0 {
      return
    }
  }
  let published = self.events.publish_token_status(
    token_id~,
    position=try! publication.position(),
    lane.output,
    source_offset=0,
    length=step.copied_bytes(),
  )
  match published {
    LunaEventPublishReady =>
      self.mark_multi_event_published(lane_index, TokenEvent)
    LunaEventPublishPayloadInvalid => {
      lane.string_cut = NoStringCut
      self.multi_begin_output_failure(lane_index)
    }
    LunaEventPublishEpochExhausted | LunaEventPublishCreditUnavailable =>
      abort("Luna online multi event publication invariant")
  }
}

///|
fn LunaOnlineInstance::multi_consume_publication(
  self : LunaOnlineInstance,
  lane_index : Int,
) -> Unit raise LunaOnlineInstanceError {
  let lane = self.multi.lanes[lane_index]
  let publication = self.lease_owner().take_publication() catch {
      _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
    }
  if !self
    .multi_worker_request(lane_index)
    .matches_publication_route(publication.route()) {
    abort("Luna online multi publication route invariant")
  }
  match publication.kind() {
    OnlineGeneratedTokenPublication =>
      self.multi_decode_token(lane_index, publication)
    OnlineSuppressedTokenPublication => lane.output_tokens += 1
    OnlineTerminalPublication => {
      let reason = try! publication.terminal_reason()
      lane.cached_input_tokens = try! publication.cached_input_tokens()
      if lane.string_cut != NoStringCut {
        if lane.string_cut == CancelledStringCut &&
          reason != OnlineCancelledByCaller {
          abort("Luna online multi string cancellation invariant")
        }
        if lane.string_cut == NaturalStringCut &&
          reason != OnlineStopTokenReached &&
          reason != OnlineMaximumOutputReached {
          abort("Luna online multi natural string invariant")
        }
        self.multi_prepare_string_terminal(lane_index)
      } else if lane.terminal_cut != NoTerminalCut {
        if lane.terminal_cut == CallerCancellationCut &&
          reason != OnlineCancelledByCaller {
          abort("Luna online multi caller cancellation invariant")
        }
        if lane.terminal_cut == DeadlineFailureCut &&
          reason != OnlineDeadlineReached {
          abort("Luna online multi deadline invariant")
        }
        if lane.terminal_cut == OutputCancellationCut &&
          reason != OnlineCancelledByCaller {
          abort("Luna online multi output cancellation invariant")
        }
        if lane.terminal_cut == WorkerFailureCut && reason != OnlineWorkerFailed {
          self.multi_prepare_worker_failure_race_terminal(lane_index, reason)
          return
        }
        self.multi_prepare_cut_terminal(lane_index)
      } else {
        self.multi_prepare_natural_terminal(lane_index, reason)
      }
    }
  }
}

///|
fn LunaOnlineInstance::multi_route_one_publication(
  self : LunaOnlineInstance,
) -> Bool raise LunaOnlineInstanceError {
  let lease = self.lease_owner()
  if !(try! lease.has_publication()) {
    return false
  }
  let route = lease.next_publication_route() catch {
    _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
  }
  let lane_index = self.multi.routes.resolve(route)
  if lane_index < 0 {
    raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
  }
  let lane = self.multi.lanes[lane_index]
  if lane.phase == LunaInstanceIdlePhase ||
    lane.worker_requests.length() != 1 ||
    !lane.worker_requests[0].matches_publication_route(route) {
    self.multi.routes.reject_authentication()
    raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
  }
  self.multi_consume_publication(lane_index)
  true
}

///|
fn LunaOnlineInstance::multi_progress_one_deadline(
  self : LunaOnlineInstance,
) -> Bool {
  let capacity = self.multi.lanes.length()
  let lane_index = self.multi.progress_lane
  self.multi.progress_lane = (lane_index + 1) % capacity
  let lane = self.multi.lanes[lane_index]
  if lane.phase != LunaRequestActivePhase ||
    !lane.accepted_acked ||
    lane.terminal_cut != NoTerminalCut ||
    lane.string_cut != NoStringCut {
    return false
  }
  if lane.deferred_cancel {
    lane.deferred_cancel = false
    ignore(self.multi_begin_cancel(lane_index, CallerCancellationCut))
    return true
  }
  let expiry = try! self.multi_worker_request(lane_index).expire()
  match expiry {
    OnlineDeadlinePending | OnlineDeadlineNaturalTerminal => false
    OnlineDeadlineCommitted => {
      lane.failure_kind = DeadlineLatchedFailure
      lane.terminal_cut = DeadlineFailureCut
      true
    }
  }
}

///|
fn LunaOnlineInstance::progress_multi_failure_mark(
  self : LunaOnlineInstance,
) -> Bool {
  if !self.multi.failure_mark_active {
    return false
  }
  let lane_index = self.multi.failure_mark_cursor
  if lane_index < 0 || lane_index >= self.multi.lanes.length() {
    abort("Luna online multi failure cursor invariant")
  }
  let lane = self.multi.lanes[lane_index]
  if lane.phase != LunaInstanceIdlePhase {
    lane.failure_kind = WorkerLatchedFailure
    lane.terminal_cut = WorkerFailureCut
    lane.recovery_close = true
  }
  self.multi.failure_mark_cursor += 1
  if self.multi.failure_mark_cursor < self.multi.lanes.length() {
    return false
  }
  self.multi.failure_mark_cursor = 0
  self.multi.failure_mark_active = false
  self.recovery_close = true
  true
}

///|
#warnings("-unused_value")
fn LunaOnlineInstance::progress_multi(
  self : LunaOnlineInstance,
) -> LunaOnlineInstanceProgress raise LunaOnlineInstanceError {
  self.require_owner()
  if self.events.has_event() {
    raise LunaOnlineInstanceFailed(LunaOnlineInvalid(LunaOnlineEventCredit))
  }
  if self.multi.failure_mark_active {
    return if self.progress_multi_failure_mark() {
      LunaOnlineInstanceTerminalizationRequired
    } else {
      LunaOnlineInstancePending
    }
  }
  if self.multi.active_count == 0 {
    return LunaOnlineInstancePending
  }
  if self.multi.retirement_lane >= 0 {
    return LunaOnlineInstanceRequestRetirementRequired
  }
  if self.multi_route_one_publication() {
    if self.events.has_event() {
      LunaOnlineInstanceEventReady
    } else {
      LunaOnlineInstancePending
    }
  } else {
    // A deferred caller cut or deadline commit may publish its authenticated
    // terminal here. Route it on the next owner turn; lower worker progress
    // rejects publication backpressure and must not reinterpret it as loss.
    if self.multi_progress_one_deadline() {
      return LunaOnlineInstancePending
    }
    let step = self.lease_owner().progress_status() catch {
        _ => raise LunaOnlineInstanceFailed(LunaOnlineServiceFailure)
      }
    match step {
      OnlineWorkerOutputInvalid | OnlineWorkerUnavailable => {
        self.multi.failure_mark_active = true
        self.multi.failure_mark_cursor = 0
        if self.progress_multi_failure_mark() {
          LunaOnlineInstanceTerminalizationRequired
        } else {
          LunaOnlineInstancePending
        }
      }
      OnlineWorkerAdvanced | OnlineWorkerRetryable => LunaOnlineInstancePending
    }
  }
}
