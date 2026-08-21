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
fn LunaOnlineFramedCoordinator::publish_rejection(
  self : LunaOnlineFramedCoordinator,
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
fn LunaOnlineFramedCoordinator::revoke_stalled_rejection(
  self : LunaOnlineFramedCoordinator,
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
fn LunaOnlineFramedCoordinator::enforce_rejection_stall(
  self : LunaOnlineFramedCoordinator,
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
fn LunaOnlineFramedCoordinator::discard_head_for_drain(
  self : LunaOnlineFramedCoordinator,
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
fn LunaOnlineFramedCoordinator::publish_head_failure(
  self : LunaOnlineFramedCoordinator,
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
fn LunaOnlineFramedCoordinator::dispatch_prepared(
  self : LunaOnlineFramedCoordinator,
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
  let prepared = self.prepared[0]
  let admission = self.instance.begin(prepared) catch {
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
      self.tickets.push(ticket)
      self.prepared.clear()
      self.pop_queue_head()
      self.semantic_event_pending = true
      LunaOnlineFramedCoordinatorAdvanced
    }
  }
}

///|
fn LunaOnlineFramedCoordinator::progress_preparation(
  self : LunaOnlineFramedCoordinator,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  if self.tickets.length() == 1 {
    if self.prepared.length() != 0 {
      abort("Luna coordinator active ticket/Prepared invariant")
    }
    if self.drain_requested && self.queue_count > 0 {
      self.discard_head_for_drain()
      return LunaOnlineFramedCoordinatorAdvanced
    }
    // Preserve the single ordered stream: preparation may advance behind the
    // active request, but its Ready/Failed disposition cannot preempt semantic
    // events that carry no coordinator sequence. Inspect it only after the
    // ticket retires.
    return match self.pool.progress() {
      LunaRequestPreparationPoolIdle => LunaOnlineFramedCoordinatorIdle
      LunaRequestPreparationPoolAdvanced => LunaOnlineFramedCoordinatorAdvanced
    }
  }
  if self.prepared.length() == 1 {
    if self.queue_count <= 0 {
      abort("Luna coordinator prepared queue invariant")
    }
    if self.drain_requested {
      self.discard_head_for_drain()
      return LunaOnlineFramedCoordinatorAdvanced
    }
    if self.tickets.is_empty() {
      return self.dispatch_prepared()
    }
    raise luna_coordinator_error(LunaOnlineFramedLifecycle)
  }
  if self.queue_count > 0 {
    let state = self.queue_head_work().state() catch {
        _ => {
          if self.drain_requested {
            self.work_slots[self.queue_head].clear()
            self.queue_sequences[self.queue_head] = 0UL
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
        let failure = self.queue_head_work().failure() catch {
            _ => raise luna_coordinator_error(LunaOnlineFramedPreparation)
          }
        if self.drain_requested {
          self.discard_head_for_drain()
          return LunaOnlineFramedCoordinatorAdvanced
        }
        self.publish_head_failure(failure)
        return LunaOnlineFramedCoordinatorRejectionReady
      }
      LunaRequestPreparationReady => {
        if self.drain_requested {
          self.discard_head_for_drain()
          return LunaOnlineFramedCoordinatorAdvanced
        }
        if self.tickets.is_empty() {
          return self.dispatch_prepared()
        }
      }
      LunaRequestPreparationCancelRequested => ()
      LunaRequestPreparationPending => ()
    }
  }
  match self.pool.progress() {
    LunaRequestPreparationPoolIdle => LunaOnlineFramedCoordinatorIdle
    LunaRequestPreparationPoolAdvanced => LunaOnlineFramedCoordinatorAdvanced
  }
}

///|
fn LunaOnlineFramedCoordinator::progress_instance(
  self : LunaOnlineFramedCoordinator,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  let state = self.instance.progress(self.ticket()) catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedMaintenance)
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
pub fn LunaOnlineFramedCoordinator::progress(
  self : LunaOnlineFramedCoordinator,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  if self.closed {
    return LunaOnlineFramedCoordinatorClosed
  }
  if self.rejection_epoch != 0UL {
    if self.enforce_rejection_stall() {
      return LunaOnlineFramedCoordinatorMaintenanceRequired
    }
    return LunaOnlineFramedCoordinatorRejectionReady
  }
  if self.event_ack_pending ||
    self.outbound_views.length() == 1 ||
    self.outbound_works.length() == 1 {
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
      (self.queue_count > 0 || !self.pool.is_drained()) {
      return self.progress_preparation()
    }
    return LunaOnlineFramedCoordinatorMaintenanceRequired
  }
  if self.tickets.length() == 1 && self.queue_count > 0 {
    if self.turn_pool {
      self.turn_pool = false
      return self.progress_preparation()
    }
    self.turn_pool = true
    return self.progress_instance()
  }
  if self.tickets.length() == 1 {
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
pub fn LunaOnlineFramedCoordinator::take_rejection(
  self : LunaOnlineFramedCoordinator,
) -> LunaOnlineFramedRejectionCredit raise LunaOnlineFramedCoordinatorError {
  if self.enforce_rejection_stall() {
    raise luna_coordinator_error(LunaOnlineFramedOutputStall)
  }
  if self.rejection_epoch == 0UL || self.rejection_taken {
    raise luna_coordinator_error(LunaOnlineFramedLifecycle)
  }
  self.rejection_taken = true
  { coordinator: self, epoch: self.rejection_epoch }
}

///|
fn LunaOnlineFramedRejectionCredit::require_owner(
  self : LunaOnlineFramedRejectionCredit,
) -> LunaOnlineFramedCoordinator raise LunaOnlineFramedCoordinatorError {
  let owner = self.coordinator
  if self.epoch == 0UL ||
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
