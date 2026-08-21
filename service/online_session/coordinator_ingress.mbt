///|
fn luna_coordinator_error(
  rule : LunaOnlineFramedCoordinatorRule,
) -> LunaOnlineFramedCoordinatorError {
  LunaOnlineFramedCoordinatorFailed(rule)
}

///|
fn luna_coordinator_request_error(
  error : @request_admission.RequestAdmissionError,
) -> LunaOnlineFramedCoordinatorError {
  match error {
    ClockUnavailable => luna_coordinator_error(LunaOnlineFramedClock)
    Invalid(LunaPreparedLifecycle) =>
      luna_coordinator_error(LunaOnlineFramedLifecycle)
    _ => luna_coordinator_error(LunaOnlineFramedPreparation)
  }
}

///|
fn luna_framed_ingress(
  kind : LunaOnlineFramedIngressKind,
  consumed : Int,
) -> LunaOnlineFramedIngress {
  {
    disposition: match kind {
      LunaOnlineFramedIngressAccepted => 0
      LunaOnlineFramedIngressBackpressured => 1
      LunaOnlineFramedIngressDraining => 2
    },
    consumed,
  }
}

///|
fn luna_framed_ingress_consumed(consumed : Int) -> LunaOnlineFramedIngress {
  luna_framed_ingress(
    if consumed > 0 {
      LunaOnlineFramedIngressAccepted
    } else {
      LunaOnlineFramedIngressBackpressured
    },
    consumed,
  )
}

///|
pub fn LunaOnlineFramedIngress::kind(
  self : LunaOnlineFramedIngress,
) -> LunaOnlineFramedIngressKind {
  match self.disposition {
    0 => LunaOnlineFramedIngressAccepted
    1 => LunaOnlineFramedIngressBackpressured
    2 => LunaOnlineFramedIngressDraining
    _ => abort("Luna framed ingress invariant")
  }
}

///|
pub fn LunaOnlineFramedIngress::consumed_bytes(
  self : LunaOnlineFramedIngress,
) -> Int {
  self.consumed
}

///|
fn LunaOnlineFramedCoordinator::queue_tail(
  self : LunaOnlineFramedCoordinator,
) -> Int {
  (self.queue_head + self.queue_count) % self.work_slots.length()
}

///|
fn LunaOnlineFramedCoordinator::queue_head_work(
  self : LunaOnlineFramedCoordinator,
) -> @request_admission.LunaRequestPreparationWork {
  if self.queue_count <= 0 || self.work_slots[self.queue_head].length() != 1 {
    abort("Luna coordinator preparation queue invariant")
  }
  self.work_slots[self.queue_head][0]
}

///|
fn LunaOnlineFramedCoordinator::pop_queue_head(
  self : LunaOnlineFramedCoordinator,
) -> Unit {
  if self.queue_count <= 0 || self.work_slots[self.queue_head].length() != 1 {
    abort("Luna coordinator preparation pop invariant")
  }
  self.work_slots[self.queue_head].clear()
  self.queue_sequences[self.queue_head] = 0UL
  self.queue_head = (self.queue_head + 1) % self.work_slots.length()
  self.queue_count -= 1
}

///|
fn LunaOnlineFramedCoordinator::current_receipt_work(
  self : LunaOnlineFramedCoordinator,
) -> @request_admission.LunaRequestPreparationWork {
  if self.receipt_slot < 0 ||
    self.receipt_slot >= self.work_slots.length() ||
    self.work_slots[self.receipt_slot].length() != 1 {
    abort("Luna coordinator receipt invariant")
  }
  self.work_slots[self.receipt_slot][0]
}

///|
fn LunaOnlineFramedCoordinator::observe_receipt_completion(
  self : LunaOnlineFramedCoordinator,
  work : @request_admission.LunaRequestPreparationWork,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  let complete = work.luna_framed_receipt_complete() catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedPreparation)
  }
  if complete {
    self.receipt_slot = -1
  }
}

///|
/// Offers one caller-owned byte range. A short accepted count leaves the tail
/// with the caller; it is never interpreted as part of the next frame.
pub fn LunaOnlineFramedCoordinator::offer_luna_framed(
  self : LunaOnlineFramedCoordinator,
  source : FixedArray[Byte],
  source_offset~ : Int,
  length~ : Int,
) -> LunaOnlineFramedIngress raise LunaOnlineFramedCoordinatorError {
  if self.closed || self.drain_requested || self.disconnect_requested {
    return luna_framed_ingress(LunaOnlineFramedIngressDraining, 0)
  }
  if self.rejection_epoch != 0UL {
    return luna_framed_ingress(LunaOnlineFramedIngressBackpressured, 0)
  }
  if self.receipt_slot < 0 && self.pool.free_lane_count() == 0 {
    return luna_framed_ingress(LunaOnlineFramedIngressBackpressured, 0)
  }
  if length < 0 || source_offset < 0 || source_offset > source.length() - length {
    raise luna_coordinator_error(LunaOnlineFramedIngress)
  }
  if self.receipt_slot >= 0 {
    let work = self.current_receipt_work()
    let consumed = work.offer_luna_framed(source, source_offset~, length~) catch {
      error => raise luna_coordinator_request_error(error)
    }
    self.observe_receipt_completion(work)
    return luna_framed_ingress_consumed(consumed)
  }
  if length == 0 {
    raise luna_coordinator_error(LunaOnlineFramedIngress)
  }
  if self.next_sequence == 0xffffffffffffffffUL {
    self.begin_drain()
    return luna_framed_ingress(LunaOnlineFramedIngressDraining, 0)
  }
  let admission = self.pool.try_begin_luna_framed(
    source,
    source_offset~,
    length~,
  ) catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedPreparation)
  }
  match admission.kind() {
    LunaRequestPreparationSaturated =>
      luna_framed_ingress(LunaOnlineFramedIngressBackpressured, 0)
    LunaRequestPreparationDraining => {
      self.begin_drain()
      luna_framed_ingress(LunaOnlineFramedIngressDraining, 0)
    }
    LunaRequestPreparationAdmitted => {
      if self.queue_count >= self.work_slots.length() {
        abort("Luna coordinator queue capacity invariant")
      }
      let slot = self.queue_tail()
      let work = admission.work() catch {
        _ => raise luna_coordinator_error(LunaOnlineFramedPreparation)
      }
      self.next_sequence += 1UL
      self.work_slots[slot].push(work)
      self.queue_sequences[slot] = self.next_sequence
      self.queue_count += 1
      self.receipt_slot = slot
      self.observe_receipt_completion(work)
      luna_framed_ingress_consumed(admission.consumed_bytes())
    }
  }
}
