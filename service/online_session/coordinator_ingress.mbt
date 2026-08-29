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
  route_sequence? : UInt64 = 0UL,
) -> LunaOnlineFramedIngress {
  {
    disposition: match kind {
      LunaOnlineFramedIngressAccepted => 0
      LunaOnlineFramedIngressBackpressured => 1
      LunaOnlineFramedIngressDraining => 2
    },
    consumed,
    route_sequence,
  }
}

///|
fn luna_framed_ingress_consumed(
  consumed : Int,
  route_sequence : UInt64,
) -> LunaOnlineFramedIngress {
  luna_framed_ingress(
    if consumed > 0 {
      LunaOnlineFramedIngressAccepted
    } else {
      LunaOnlineFramedIngressBackpressured
    },
    consumed,
    route_sequence~,
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
/// Reports whether this transition belongs to a newly admitted or continuing
/// framed submission. Backpressure before admission and drain have no route.
pub fn LunaOnlineFramedIngress::has_request_route(
  self : LunaOnlineFramedIngress,
) -> Bool {
  self.route_sequence != 0UL
}

///|
/// Returns the exact opaque framed-submission route carried by this ingress.
pub fn LunaOnlineFramedIngress::request_route(
  self : LunaOnlineFramedIngress,
) -> LunaOnlineFramedRequestRoute raise LunaOnlineFramedCoordinatorError {
  if self.route_sequence == 0UL {
    raise luna_coordinator_error(LunaOnlineFramedOwnership)
  }
  { sequence: self.route_sequence }
}

///|
/// Compares two inert route identities without exposing their scalar encoding.
pub fn LunaOnlineFramedRequestRoute::same_request(
  self : LunaOnlineFramedRequestRoute,
  other : LunaOnlineFramedRequestRoute,
) -> Bool {
  self.sequence != 0UL && self.sequence == other.sequence
}

///|
fn LunaOnlineFramedService::queue_tail(self : LunaOnlineFramedService) -> Int {
  (self.queue_head + self.queue_count) % self.work_slots.length()
}

///|
fn LunaOnlineFramedService::queue_head_work(
  self : LunaOnlineFramedService,
) -> @request_admission.LunaRequestPreparationWork {
  if self.queue_count <= 0 || self.work_slots[self.queue_head].length() != 1 {
    abort("Luna coordinator preparation queue invariant")
  }
  self.work_slots[self.queue_head][0]
}

///|
fn LunaOnlineFramedService::pop_queue_head(
  self : LunaOnlineFramedService,
) -> Unit {
  if self.queue_count <= 0 || self.work_slots[self.queue_head].length() != 1 {
    abort("Luna coordinator preparation pop invariant")
  }
  self.work_slots[self.queue_head].clear()
  self.queue_sequences[self.queue_head] = 0UL
  self.queue_abandoned[self.queue_head] = false
  self.queue_head = (self.queue_head + 1) % self.work_slots.length()
  self.queue_count -= 1
}

///|
fn LunaOnlineFramedService::current_receipt_work(
  self : LunaOnlineFramedService,
) -> @request_admission.LunaRequestPreparationWork {
  if self.receipt_slot < 0 ||
    self.receipt_slot >= self.work_slots.length() ||
    self.work_slots[self.receipt_slot].length() != 1 {
    abort("Luna coordinator receipt invariant")
  }
  self.work_slots[self.receipt_slot][0]
}

///|
fn LunaOnlineFramedService::observe_receipt_completion(
  self : LunaOnlineFramedService,
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
fn LunaOnlineFramedService::stream_offer_luna_framed(
  self : LunaOnlineFramedService,
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
    let route_sequence = self.queue_sequences[self.receipt_slot]
    let work = self.current_receipt_work()
    let consumed = work.offer_luna_framed(source, source_offset~, length~) catch {
      error => raise luna_coordinator_request_error(error)
    }
    self.observe_receipt_completion(work)
    return luna_framed_ingress_consumed(consumed, route_sequence)
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
      self.queue_abandoned[slot] = false
      self.queue_count += 1
      self.receipt_slot = slot
      self.observe_receipt_completion(work)
      luna_framed_ingress_consumed(
        admission.consumed_bytes(),
        self.next_sequence,
      )
    }
  }
}

///|
fn LunaOnlineFramedService::stream_offer_luna_framed_with_receipt(
  self : LunaOnlineFramedService,
  receipt : @request_admission.LunaRequestReceipt,
  source : FixedArray[Byte],
  source_offset~ : Int,
  length~ : Int,
) -> LunaOnlineFramedIngress raise LunaOnlineFramedCoordinatorError {
  if self.closed || self.drain_requested || self.disconnect_requested {
    return luna_framed_ingress(LunaOnlineFramedIngressDraining, 0)
  }
  if self.rejection_epoch != 0UL || self.receipt_slot >= 0 {
    return luna_framed_ingress(LunaOnlineFramedIngressBackpressured, 0)
  }
  if self.pool.free_lane_count() == 0 {
    return luna_framed_ingress(LunaOnlineFramedIngressBackpressured, 0)
  }
  if self.next_sequence == 0xffffffffffffffffUL {
    self.begin_drain()
    return luna_framed_ingress(LunaOnlineFramedIngressDraining, 0)
  }
  let admission = self.pool.try_begin_luna_framed_with_receipt(
    receipt,
    source,
    source_offset~,
    length~,
  ) catch {
    error => raise luna_coordinator_request_error(error)
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
      self.queue_abandoned[slot] = false
      self.queue_count += 1
      self.receipt_slot = slot
      self.observe_receipt_completion(work)
      luna_framed_ingress_consumed(
        admission.consumed_bytes(),
        self.next_sequence,
      )
    }
  }
}
