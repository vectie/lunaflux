///|
fn luna_online_rejection_rule(
  error : @request_admission.RequestAdmissionError,
) -> LunaOnlineFramedRejectionRule {
  match error {
    ClockUnavailable => LunaOnlineFramedRejectedDeadline
    Invalid(rule) =>
      match rule {
        ModelIdentity | TokenizerIdentity => LunaOnlineFramedRejectedIdentity
        RequestContract | FrameContract | ReceiptTime =>
          LunaOnlineFramedRejectedFrame
        Deadline => LunaOnlineFramedRejectedDeadline
        Input => LunaOnlineFramedRejectedInput
        Tokenization => LunaOnlineFramedRejectedTokenization
        LunaPreparationPool | LunaPreparationStorage =>
          LunaOnlineFramedRejectedCapacity
        SchedulerContract
        | OutputContract
        | LunaPreparedLifecycle
        | LunaPreparationWork => LunaOnlineFramedRejectedService
      }
  }
}

///|
fn luna_rejection_rule_code(rule : LunaOnlineFramedRejectionRule) -> Int {
  match rule {
    LunaOnlineFramedRejectedFrame => 1
    LunaOnlineFramedRejectedIdentity => 2
    LunaOnlineFramedRejectedDeadline => 3
    LunaOnlineFramedRejectedInput => 4
    LunaOnlineFramedRejectedTokenization => 5
    LunaOnlineFramedRejectedCapacity => 6
    LunaOnlineFramedRejectedService => 7
  }
}

///|
fn LunaOnlineFramedService::publish_rejection(
  self : LunaOnlineFramedService,
  sequence : UInt64,
  rule : LunaOnlineFramedRejectionRule,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  if self.rejection_epoch != 0UL {
    raise luna_coordinator_error(LunaOnlineFramedLifecycle)
  }
  if self.next_sequence == 0UL || sequence == 0UL {
    abort("Luna coordinator rejection sequence invariant")
  }
  if self.rejection_epoch == 0xffffffffffffffffUL {
    self.begin_drain()
    raise luna_coordinator_error(LunaOnlineFramedOwnership)
  }
  // Epoch zero means no pinned rejection. Since acknowledgements also clear
  // it, use the request sequence as the nonzero exact result epoch.
  self.rejection_epoch = sequence
  self.rejection_sequence = sequence
  self.rejection_rule = luna_rejection_rule_code(rule)
  self.rejection_taken = false
  if !self.begin_stall_deadline() {
    self.revoke_stalled_rejection()
    raise luna_coordinator_error(LunaOnlineFramedClock)
  }
}

///|
fn LunaOnlineFramedService::revoke_stalled_rejection(
  self : LunaOnlineFramedService,
) -> Unit {
  self.rejection_epoch = 0UL
  self.rejection_sequence = 0UL
  self.rejection_rule = 0
  self.rejection_taken = false
  self.stall_active = false
  self.stall_deadline_millis = 0UL
  self.begin_output_disconnect() catch {
    _ => ()
  }
}

///|
fn LunaOnlineFramedService::enforce_rejection_stall(
  self : LunaOnlineFramedService,
) -> Bool {
  if self.rejection_epoch == 0UL {
    return false
  }
  if self.stall_expired() == 0 {
    return false
  }
  self.revoke_stalled_rejection()
  true
}

///|
fn LunaOnlineFramedService::discard_head_for_drain(
  self : LunaOnlineFramedService,
) -> Unit {
  let head = self.queue_head
  if self.prepared.length() == 1 {
    try! self.prepared[0].discard()
    self.prepared.clear()
  } else {
    try! self.queue_head_work().discard()
  }
  if self.receipt_slot == head {
    self.receipt_slot = -1
  }
  self.pop_queue_head()
}

///|
fn LunaOnlineFramedService::publish_head_failure(
  self : LunaOnlineFramedService,
  error : @request_admission.RequestAdmissionError,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  let head = self.queue_head
  let sequence = self.queue_sequences[head]
  let work = self.queue_head_work()
  work.discard() catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedPreparation)
  }
  if self.receipt_slot == head {
    self.receipt_slot = -1
  }
  self.pop_queue_head()
  self.publish_rejection(sequence, luna_online_rejection_rule(error))
}

///|
fn LunaOnlineFramedService::dispatch_prepared(
  self : LunaOnlineFramedService,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  let sequence = self.queue_sequences[self.queue_head]
  if self.prepared.is_empty() {
    let work = self.queue_head_work()
    let prepared = work.take_prepared() catch {
      error => {
        let failure = work.failure() catch { _ => error }
        self.publish_head_failure(failure)
        return LunaOnlineFramedCoordinatorRejectionReady
      }
    }
    self.prepared.push(prepared)
  }
  if self.queue_abandoned[self.queue_head] {
    self.prepared[0].discard() catch {
      _ => raise luna_coordinator_error(LunaOnlineFramedPreparation)
    }
    self.prepared.clear()
    self.pop_queue_head()
    return LunaOnlineFramedCoordinatorAdvanced
  }
  let prepared = self.prepared[0]
  let admission = self.instance.begin_multi(prepared) catch {
    _ => {
      // Preflight errors leave Prepared live; lower-admission errors consume
      // it after releasing all lower authority. Either disposition is safe to
      // probe through the one-shot discard boundary.
      prepared.discard() catch {
        _ => ()
      }
      self.prepared.clear()
      self.pop_queue_head()
      self.publish_rejection(sequence, LunaOnlineFramedRejectedService)
      return LunaOnlineFramedCoordinatorRejectionReady
    }
  }
  match admission.kind() {
    LunaOnlineInstanceBusy => LunaOnlineFramedCoordinatorIdle
    LunaOnlineInstanceDraining => {
      prepared.discard() catch {
        _ => raise luna_coordinator_error(LunaOnlineFramedPreparation)
      }
      self.prepared.clear()
      self.pop_queue_head()
      self.begin_drain()
      LunaOnlineFramedCoordinatorDraining
    }
    LunaOnlineInstanceAdmitted => {
      let ticket = admission.ticket() catch {
        _ => raise luna_coordinator_error(LunaOnlineFramedLifecycle)
      }
      let was_empty = self.tickets.is_empty()
      self.tickets.push(ticket)
      self.ticket_sequences.push(sequence)
      self.begin_request_timing(ticket)
      if was_empty {
        self.worker_failure_observed = false
      }
      self.prepared.clear()
      self.pop_queue_head()
      self.semantic_event_pending = true
      LunaOnlineFramedCoordinatorAdvanced
    }
  }
}

///|
fn LunaOnlineFramedService::progress_preparation(
  self : LunaOnlineFramedService,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  if self.prepared.length() == 1 {
    if self.queue_count <= 0 {
      abort("Luna coordinator prepared queue invariant")
    }
    if self.drain_requested || self.disconnect_requested {
      self.discard_head_for_drain()
      return LunaOnlineFramedCoordinatorAdvanced
    }
    return self.dispatch_prepared()
  }
  if self.queue_count > 0 {
    if self.disconnect_requested {
      self.discard_head_for_drain()
      return LunaOnlineFramedCoordinatorAdvanced
    }
    let state = self.queue_head_work().state() catch {
        _ => {
          if self.drain_requested ||
            self.disconnect_requested ||
            self.queue_abandoned[self.queue_head] {
            self.work_slots[self.queue_head].clear()
            self.queue_sequences[self.queue_head] = 0UL
            self.queue_abandoned[self.queue_head] = false
            self.queue_head = (self.queue_head + 1) % self.work_slots.length()
            self.queue_count -= 1
            self.receipt_slot = -1
            return LunaOnlineFramedCoordinatorAdvanced
          }
          raise luna_coordinator_error(LunaOnlineFramedPreparation)
        }
      }
    match state {
      LunaRequestPreparationFailed => {
        if self.queue_abandoned[self.queue_head] {
          self.discard_head_for_drain()
          return LunaOnlineFramedCoordinatorAdvanced
        }
        let failure = self.queue_head_work().failure() catch {
            _ => raise luna_coordinator_error(LunaOnlineFramedPreparation)
          }
        if self.drain_requested || self.disconnect_requested {
          self.discard_head_for_drain()
          return LunaOnlineFramedCoordinatorAdvanced
        }
        if !self.tickets.is_empty() {
          return LunaOnlineFramedCoordinatorAdvanced
        }
        self.publish_head_failure(failure)
        return LunaOnlineFramedCoordinatorRejectionReady
      }
      LunaRequestPreparationReady => {
        if self.drain_requested || self.disconnect_requested {
          self.discard_head_for_drain()
          return LunaOnlineFramedCoordinatorAdvanced
        }
        return self.dispatch_prepared()
      }
      LunaRequestPreparationCancelRequested => ()
      LunaRequestPreparationPending => ()
    }
  }
  match self.pool.progress() {
    LunaRequestPreparationPoolIdle => LunaOnlineFramedCoordinatorIdle
    LunaRequestPreparationPoolAwaitingInput =>
      LunaOnlineFramedCoordinatorAwaitingInput
    LunaRequestPreparationPoolAdvanced => LunaOnlineFramedCoordinatorAdvanced
  }
}

///|
fn LunaOnlineFramedService::progress_instance(
  self : LunaOnlineFramedService,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  let state = self.instance.progress_multi() catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedMaintenance)
  }
  if self.publish_worker_failure_if_needed() {
    return LunaOnlineFramedCoordinatorObservationReady
  }
  match state {
    LunaOnlineInstancePending => LunaOnlineFramedCoordinatorAdvanced
    LunaOnlineInstanceEventReady => {
      self.semantic_event_pending = true
      LunaOnlineFramedCoordinatorAdvanced
    }
    LunaOnlineInstanceTerminalizationRequired => {
      self.maintenance_kind = 1
      LunaOnlineFramedCoordinatorMaintenanceRequired
    }
    LunaOnlineInstanceRequestRetirementRequired => {
      self.maintenance_kind = 2
      LunaOnlineFramedCoordinatorMaintenanceRequired
    }
  }
}

///|
/// Advances at most one preparation, online-instance, or framed-output
/// transition. Blocking recovery, retirement, and shutdown are never driven
/// here.
fn LunaOnlineFramedService::progress_reactor_core(
  self : LunaOnlineFramedService,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  if self.closed {
    return LunaOnlineFramedCoordinatorClosed
  }
  if self.observation_kind != 0 {
    return LunaOnlineFramedCoordinatorObservationReady
  }
  if self.rejection_epoch != 0UL {
    if self.enforce_rejection_stall() {
      return LunaOnlineFramedCoordinatorMaintenanceRequired
    }
    return LunaOnlineFramedCoordinatorRejectionReady
  }
  if self.disconnect_requested && !self.tickets.is_empty() {
    self.begin_output_disconnect() catch {
      _ => return LunaOnlineFramedCoordinatorMaintenanceRequired
    }
  }
  if self.event_ack_pending ||
    self.outbound_views.length() == 1 ||
    self.outbound_works.length() == 1 ||
    (self.stream_mode == 2 && self.event_credits.length() == 1) {
    return self.progress_outbound_event()
  }
  if self.semantic_event_pending {
    return self.start_semantic_event()
  }
  if self.maintenance_kind != 0 {
    return LunaOnlineFramedCoordinatorMaintenanceRequired
  }
  if self.disconnect_requested {
    // A cut stream with no admitted request still owns every queued
    // preparation lane. Keep the reactor-side proportional pool driver live
    // until those lanes have observed drain and released their authority;
    // shutdown itself remains off-reactor.
    if self.tickets.is_empty() &&
      (
        self.queue_count > 0 ||
        self.pool.free_lane_count() != self.pool.lane_count()
      ) {
      return self.progress_preparation()
    }
    return LunaOnlineFramedCoordinatorMaintenanceRequired
  }
  if !self.tickets.is_empty() && self.queue_count > 0 {
    if self.turn_pool {
      self.turn_pool = false
      return self.progress_preparation()
    }
    self.turn_pool = true
    return self.progress_instance()
  }
  if !self.tickets.is_empty() {
    return self.progress_instance()
  }
  if self.queue_count > 0 || !self.pool.is_drained() {
    return self.progress_preparation()
  }
  if self.drain_requested {
    // Preparation drain is complete; the remaining instance shutdown may
    // close/reap lower owners and must be handed to the off-reactor driver.
    return LunaOnlineFramedCoordinatorMaintenanceRequired
  }
  LunaOnlineFramedCoordinatorIdle
}

///|
fn LunaOnlineFramedService::stream_take_rejection(
  self : LunaOnlineFramedService,
) -> LunaOnlineFramedRejectionCredit raise LunaOnlineFramedCoordinatorError {
  if self.enforce_rejection_stall() {
    raise luna_coordinator_error(LunaOnlineFramedOutputStall)
  }
  if self.rejection_epoch == 0UL || self.rejection_taken {
    raise luna_coordinator_error(LunaOnlineFramedLifecycle)
  }
  self.rejection_taken = true
  {
    service: self,
    stream_epoch: self.stream_epoch,
    epoch: self.rejection_epoch,
  }
}

///|
fn LunaOnlineFramedRejectionCredit::require_owner(
  self : LunaOnlineFramedRejectionCredit,
) -> LunaOnlineFramedService raise LunaOnlineFramedCoordinatorError {
  let owner = self.service
  if self.epoch == 0UL ||
    self.stream_epoch == 0UL ||
    owner.stream_epoch != self.stream_epoch ||
    (owner.stream_phase != 2 && owner.stream_phase != 3) ||
    owner.rejection_epoch != self.epoch ||
    !owner.rejection_taken {
    raise luna_coordinator_error(LunaOnlineFramedLifecycle)
  }
  if owner.enforce_rejection_stall() {
    raise luna_coordinator_error(LunaOnlineFramedOutputStall)
  }
  owner
}

///|
pub fn LunaOnlineFramedRejectionCredit::sequence(
  self : LunaOnlineFramedRejectionCredit,
) -> UInt64 raise LunaOnlineFramedCoordinatorError {
  self.require_owner().rejection_sequence
}

///|
/// Returns the exact opaque route of this rejected framed submission.
pub fn LunaOnlineFramedRejectionCredit::request_route(
  self : LunaOnlineFramedRejectionCredit,
) -> LunaOnlineFramedRequestRoute raise LunaOnlineFramedCoordinatorError {
  let sequence = self.require_owner().rejection_sequence
  if sequence == 0UL {
    abort("Luna coordinator rejection route invariant")
  }
  { sequence, }
}

///|
pub fn LunaOnlineFramedRejectionCredit::rule(
  self : LunaOnlineFramedRejectionCredit,
) -> LunaOnlineFramedRejectionRule raise LunaOnlineFramedCoordinatorError {
  match self.require_owner().rejection_rule {
    1 => LunaOnlineFramedRejectedFrame
    2 => LunaOnlineFramedRejectedIdentity
    3 => LunaOnlineFramedRejectedDeadline
    4 => LunaOnlineFramedRejectedInput
    5 => LunaOnlineFramedRejectedTokenization
    6 => LunaOnlineFramedRejectedCapacity
    7 => LunaOnlineFramedRejectedService
    _ => abort("Luna coordinator rejection rule invariant")
  }
}

///|
pub fn LunaOnlineFramedRejectionCredit::ack(
  self : LunaOnlineFramedRejectionCredit,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  let owner = self.require_owner()
  owner.rejection_epoch = 0UL
  owner.rejection_sequence = 0UL
  owner.rejection_rule = 0
  owner.rejection_taken = false
  owner.stall_active = false
  owner.stall_deadline_millis = 0UL
}
