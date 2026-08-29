///|
fn LunaOnlineFramedService::stream_offer_luna_text_handoff_with_receipt(
  self : LunaOnlineFramedService,
  receipt : @request_admission.LunaRequestReceipt,
  source : @inference.LunaTextRequestHandoffLease,
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
  let admission = self.pool.try_begin_luna_text_handoff_with_receipt(
    receipt, source,
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
        abort("Luna typed coordinator queue capacity invariant")
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
      luna_framed_ingress_consumed(1, self.next_sequence)
    }
  }
}

///|
pub fn LunaOnlineFramedStream::offer_luna_text_handoff_with_receipt(
  self : LunaOnlineFramedStream,
  receipt : @request_admission.LunaRequestReceipt,
  source : @inference.LunaTextRequestHandoffLease,
) -> LunaOnlineFramedIngress raise LunaOnlineFramedCoordinatorError {
  self
  .require_service()
  .stream_offer_luna_text_handoff_with_receipt(receipt, source)
}
