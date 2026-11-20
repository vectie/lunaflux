///|
fn LunaOnlineFramedService::stream_framed_event_remaining(
  self : LunaOnlineFramedService,
) -> Int32 raise LunaOnlineFramedCoordinatorError {
  if self.stream_mode != 1 {
    raise luna_coordinator_error(LunaOnlineFramedOwnership)
  }
  if self.enforce_output_stall() != 0 {
    raise luna_coordinator_error(LunaOnlineFramedOutputStall)
  }
  if self.outbound_views.length() != 1 {
    raise luna_coordinator_error(LunaOnlineFramedEvent)
  }
  self.event_length - self.confirmed_bytes
}

///|
fn LunaOnlineFramedService::stream_copy_framed_event_chunk_to(
  self : LunaOnlineFramedService,
  destination : FixedArray[Byte],
  destination_offset~ : Int32,
  maximum_length~ : Int32,
) -> LunaOnlineFramedEventOffer raise LunaOnlineFramedCoordinatorError {
  if self.stream_mode != 1 {
    raise luna_coordinator_error(LunaOnlineFramedOwnership)
  }
  if self.enforce_output_stall() != 0 {
    raise luna_coordinator_error(LunaOnlineFramedOutputStall)
  }
  if self.outbound_views.length() != 1 || self.offered_bytes != 0 {
    raise luna_coordinator_error(LunaOnlineFramedEvent)
  }
  let remaining = self.event_length - self.confirmed_bytes
  if maximum_length <= 0 || remaining <= 0 {
    raise luna_coordinator_error(LunaOnlineFramedEvent)
  }
  if self.offer_epoch == 0xffffffffffffffffUL {
    self.begin_output_disconnect() catch {
      _ => ()
    }
    raise luna_coordinator_error(LunaOnlineFramedOwnership)
  }
  let amount = Int32::min(
    Int32::min(maximum_length, self.event_copy_quantum),
    remaining,
  )
  ignore(
    self.outbound_views[0].copy_chunk_to(
      destination,
      destination_offset~,
      source_offset=self.confirmed_bytes,
      length=amount,
    ) catch {
      _ => raise luna_coordinator_error(LunaOnlineFramedEvent)
    },
  )
  self.offered_bytes = amount
  self.offer_epoch += 1UL
  {
    service: self,
    stream_epoch: self.stream_epoch,
    event_epoch: self.event_epoch,
    offer_epoch: self.offer_epoch,
    length: amount,
  }
}

///|
fn LunaOnlineFramedEventOffer::require_owner(
  self : LunaOnlineFramedEventOffer,
) -> LunaOnlineFramedService raise LunaOnlineFramedCoordinatorError {
  let owner = self.service
  if self.event_epoch == 0UL ||
    self.stream_epoch == 0UL ||
    owner.stream_epoch != self.stream_epoch ||
    (owner.stream_phase != 2 && owner.stream_phase != 3) ||
    owner.stream_mode != 1 ||
    self.offer_epoch == 0UL ||
    owner.event_epoch != self.event_epoch ||
    owner.offer_epoch != self.offer_epoch ||
    owner.offered_bytes != self.length ||
    owner.outbound_views.length() != 1 {
    raise luna_coordinator_error(LunaOnlineFramedEvent)
  }
  owner
}

///|
pub fn LunaOnlineFramedEventOffer::length(
  self : LunaOnlineFramedEventOffer,
) -> Int32 raise LunaOnlineFramedCoordinatorError {
  ignore(self.require_owner())
  self.length
}

///|
pub fn LunaOnlineFramedEventOffer::confirm(
  self : LunaOnlineFramedEventOffer,
  length~ : Int32,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  let owner = self.require_owner()
  if owner.enforce_output_stall() != 0 {
    raise luna_coordinator_error(LunaOnlineFramedOutputStall)
  }
  if length <= 0 || length > self.length {
    raise luna_coordinator_error(LunaOnlineFramedEvent)
  }
  let next_confirmed = owner.confirmed_bytes + length
  if next_confirmed < owner.event_length {
    owner.confirmed_bytes = next_confirmed
    owner.offered_bytes = 0
    return LunaOnlineFramedCoordinatorEventReady
  }
  // Exact authentication above proves this is the sole live framed View.
  // Release before invalidating the Offer or mutating confirmed cursors.
  try! owner.outbound_views[0].release()
  owner.outbound_views.clear()
  owner.offered_bytes = 0
  owner.event_length = 0
  owner.confirmed_bytes = 0
  owner.event_ack_pending = true
  if owner.event_kind == 3 {
    owner.maintenance_kind = 3
    return LunaOnlineFramedCoordinatorMaintenanceRequired
  }
  owner.finish_event_ack()
}
