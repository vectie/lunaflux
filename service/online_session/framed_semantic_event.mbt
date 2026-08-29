///|
fn LunaOnlineFramedService::stream_take_semantic_event(
  self : LunaOnlineFramedService,
) -> LunaOnlineFramedSemanticEvent raise LunaOnlineFramedCoordinatorError {
  if self.stream_mode != 2 ||
    self.event_credits.length() != 1 ||
    self.semantic_event_taken ||
    self.event_ack_pending ||
    self.outbound_works.length() != 0 ||
    self.outbound_views.length() != 0 {
    raise luna_coordinator_error(LunaOnlineFramedOwnership)
  }
  if self.enforce_output_stall() != 0 {
    raise luna_coordinator_error(LunaOnlineFramedOutputStall)
  }
  self.semantic_event_taken = true
  {
    service: self,
    stream_epoch: self.stream_epoch,
    event_epoch: self.event_epoch,
    route_sequence: self.event_route_sequence,
  }
}

///|
fn LunaOnlineFramedSemanticEvent::require_owner(
  self : LunaOnlineFramedSemanticEvent,
) -> LunaOnlineFramedService raise LunaOnlineFramedCoordinatorError {
  let owner = self.service
  if self.stream_epoch == 0UL ||
    self.event_epoch == 0UL ||
    owner.stream_epoch != self.stream_epoch ||
    (owner.stream_phase != 2 && owner.stream_phase != 3) ||
    owner.stream_mode != 2 ||
    owner.event_epoch != self.event_epoch ||
    self.route_sequence == 0UL ||
    owner.event_route_sequence != self.route_sequence ||
    owner.event_credits.length() != 1 ||
    !owner.semantic_event_taken ||
    owner.event_ack_pending ||
    owner.outbound_works.length() != 0 ||
    owner.outbound_views.length() != 0 {
    raise luna_coordinator_error(LunaOnlineFramedOwnership)
  }
  owner
}

///|
/// Returns the exact opaque framed-submission route for this semantic event.
pub fn LunaOnlineFramedSemanticEvent::request_route(
  self : LunaOnlineFramedSemanticEvent,
) -> LunaOnlineFramedRequestRoute raise LunaOnlineFramedCoordinatorError {
  ignore(self.require_owner())
  { sequence: self.route_sequence }
}

///|
/// Borrows the exact semantic event view. The service retains acknowledgement
/// and disconnect authority; advancing or retiring the event stales the view.
pub fn LunaOnlineFramedSemanticEvent::view(
  self : LunaOnlineFramedSemanticEvent,
) -> @luna_event.LunaEventView raise LunaOnlineFramedCoordinatorError {
  let owner = self.require_owner()
  if owner.enforce_output_stall() != 0 {
    raise luna_coordinator_error(LunaOnlineFramedOutputStall)
  }
  owner.event_credits[0].view() catch {
    _ => raise luna_coordinator_error(LunaOnlineFramedEvent)
  }
}

///|
/// Marks this exact event as delivered. A later stream progress transition
/// performs the lower acknowledgement; this call never acknowledges inline.
pub fn LunaOnlineFramedSemanticEvent::delivered(
  self : LunaOnlineFramedSemanticEvent,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  let owner = self.require_owner()
  if owner.enforce_output_stall() != 0 {
    raise luna_coordinator_error(LunaOnlineFramedOutputStall)
  }
  owner.semantic_event_taken = false
  owner.event_ack_pending = true
}

///|
/// Cancels the exact routed request and discards this event without claiming
/// transport delivery. The lower acknowledgement remains a later coordinator
/// transition, exactly as for successful delivery.
pub fn LunaOnlineFramedSemanticEvent::cancel_and_discard(
  self : LunaOnlineFramedSemanticEvent,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  let owner = self.require_owner()
  owner.request_cancel_route({ sequence: self.route_sequence })
  owner.semantic_event_taken = false
  owner.event_ack_pending = true
}

///|
/// Discards this exact event after its transport route has already been cut.
/// This does not assert delivery and does not create cancellation authority.
pub fn LunaOnlineFramedSemanticEvent::discard_after_transport_cut(
  self : LunaOnlineFramedSemanticEvent,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  let owner = self.require_owner()
  owner.semantic_event_taken = false
  owner.event_ack_pending = true
}
