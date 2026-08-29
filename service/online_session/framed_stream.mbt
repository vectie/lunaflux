///|
fn LunaOnlineFramedService::record_cooperative_progress(
  self : LunaOnlineFramedService,
  progress : LunaOnlineFramedCoordinatorProgress,
) -> LunaOnlineFramedCoordinatorProgress {
  self.cooperative_handoff = progress ==
    LunaOnlineFramedCoordinatorMaintenanceRequired
  progress
}

///|
fn LunaOnlineFramedService::progress_cooperatively(
  self : LunaOnlineFramedService,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  let progress = if self.cooperative_handoff {
    self.cooperative_handoff = false
    self.progress_maintenance_core()
  } else {
    self.progress_reactor_core()
  }
  if self.stream_phase == 3 && self.finish_stream_retirement_if_ready() {
    return LunaOnlineFramedCoordinatorAdvanced
  }
  self.record_cooperative_progress(progress)
}

///|
pub fn LunaOnlineFramedStream::offer_luna_framed(
  self : LunaOnlineFramedStream,
  source : FixedArray[Byte],
  source_offset~ : Int,
  length~ : Int,
) -> LunaOnlineFramedIngress raise LunaOnlineFramedCoordinatorError {
  self
  .require_service()
  .stream_offer_luna_framed(source, source_offset~, length~)
}

///|
/// Transfers an independently captured first-byte receipt only when this
/// exact stream obtains a preparation lane. Later chunks use `offer_luna_framed`.
pub fn LunaOnlineFramedStream::offer_luna_framed_with_receipt(
  self : LunaOnlineFramedStream,
  receipt : @request_admission.LunaRequestReceipt,
  source : FixedArray[Byte],
  source_offset~ : Int,
  length~ : Int,
) -> LunaOnlineFramedIngress raise LunaOnlineFramedCoordinatorError {
  self
  .require_service()
  .stream_offer_luna_framed_with_receipt(
    receipt,
    source,
    source_offset~,
    length~,
  )
}

///|
pub fn LunaOnlineFramedStream::transport_wait_remaining_millis(
  self : LunaOnlineFramedStream,
) -> Int raise LunaOnlineFramedCoordinatorError {
  self.require_service().stream_transport_wait_remaining_millis()
}

///|
pub fn LunaOnlineFramedStream::maintenance_wait_remaining_millis(
  self : LunaOnlineFramedStream,
) -> Int raise LunaOnlineFramedCoordinatorError {
  self.require_service().maintenance_wait_remaining_millis()
}

///|
/// Reports whether the exact current direct frame receipt has completed. The
/// scalar carries no preparation Work, lane, receipt timestamp, or byte view.
pub fn LunaOnlineFramedStream::luna_framed_receipt_complete(
  self : LunaOnlineFramedStream,
) -> Bool raise LunaOnlineFramedCoordinatorError {
  let service = self.require_service()
  service.receipt_slot < 0 && service.queue_count > 0
}

///|
/// Reports whether no incomplete direct frame receipt is retained. The scalar
/// carries no Work, lane, receipt timestamp, request identity, or byte view.
pub fn LunaOnlineFramedStream::luna_framed_input_clear(
  self : LunaOnlineFramedStream,
) -> Bool raise LunaOnlineFramedCoordinatorError {
  self.require_service().receipt_slot < 0
}

///|
/// Reports whether the coordinator retains no incomplete receipt, queued or
/// prepared request, or active ticket. The scalar carries no Work, lane,
/// receipt timestamp, request identity, or byte view.
pub fn LunaOnlineFramedStream::luna_framed_boundary_clear(
  self : LunaOnlineFramedStream,
) -> Bool raise LunaOnlineFramedCoordinatorError {
  let service = self.require_service()
  service.receipt_slot < 0 &&
  service.queue_count == 0 &&
  service.prepared.is_empty() &&
  service.tickets.is_empty() &&
  service.ticket_sequences.is_empty() &&
  service.pool.free_lane_count() == service.pool.lane_count() &&
  service.event_credits.is_empty() &&
  service.outbound_works.is_empty() &&
  service.outbound_views.is_empty() &&
  service.event_kind == 0 &&
  service.event_length == 0 &&
  service.confirmed_bytes == 0 &&
  service.offered_bytes == 0 &&
  !service.semantic_event_pending &&
  !service.event_ack_pending &&
  service.event_route_sequence == 0UL &&
  service.maintenance_kind == 0 &&
  service.rejection_epoch == 0UL &&
  service.observation_kind == 0
}

///|
pub fn LunaOnlineFramedStream::request_cancel(
  self : LunaOnlineFramedStream,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  self.require_service().request_cancel()
}

///|
/// Requests cancellation for exactly the admitted request represented by the
/// opaque route. Stale, foreign, and already-retired routes fail closed.
pub fn LunaOnlineFramedStream::request_cancel_route(
  self : LunaOnlineFramedStream,
  route : LunaOnlineFramedRequestRoute,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  self.require_service().request_cancel_route(route)
}

///|
/// Reports whether the exact opaque route still names an admitted request.
/// It exposes no lane, generation, scheduler ticket, or mutable authority.
pub fn LunaOnlineFramedStream::request_route_live(
  self : LunaOnlineFramedStream,
  route : LunaOnlineFramedRequestRoute,
) -> Bool raise LunaOnlineFramedCoordinatorError {
  self.require_service().request_route_live(route)
}

///|
pub fn LunaOnlineFramedStream::disconnect(
  self : LunaOnlineFramedStream,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  self.require_service().begin_stream_disconnect()
}

///|
pub fn LunaOnlineFramedStream::progress(
  self : LunaOnlineFramedStream,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  self.require_service().progress_cooperatively()
}

///|
pub fn LunaOnlineFramedStream::framed_event_remaining(
  self : LunaOnlineFramedStream,
) -> Int raise LunaOnlineFramedCoordinatorError {
  self.require_service().stream_framed_event_remaining()
}

///|
pub fn LunaOnlineFramedStream::copy_framed_event_chunk_to(
  self : LunaOnlineFramedStream,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
  maximum_length~ : Int,
) -> LunaOnlineFramedEventOffer raise LunaOnlineFramedCoordinatorError {
  self
  .require_service()
  .stream_copy_framed_event_chunk_to(
    destination,
    destination_offset~,
    maximum_length~,
  )
}

///|
/// Takes the one exact semantic event delivery capability published for this
/// semantic-mode stream. Native framed streams cannot take semantic events.
pub fn LunaOnlineFramedStream::take_semantic_event(
  self : LunaOnlineFramedStream,
) -> LunaOnlineFramedSemanticEvent raise LunaOnlineFramedCoordinatorError {
  self.require_service().stream_take_semantic_event()
}

///|
/// Takes the exact finite observation currently backpressuring this stream.
pub fn LunaOnlineFramedStream::take_observation(
  self : LunaOnlineFramedStream,
) -> LunaOnlineFramedObservation raise LunaOnlineFramedCoordinatorError {
  self.require_service().stream_take_observation()
}

///|
pub fn LunaOnlineFramedStream::take_rejection(
  self : LunaOnlineFramedStream,
) -> LunaOnlineFramedRejectionCredit raise LunaOnlineFramedCoordinatorError {
  self.require_service().stream_take_rejection()
}

///|
pub fn LunaOnlineFramedService::progress(
  self : LunaOnlineFramedService,
) -> LunaOnlineFramedServiceReadiness raise LunaOnlineFramedCoordinatorError {
  if self.closed {
    return LunaOnlineFramedServiceClosed
  }
  if self.stream_phase == 1 || self.stream_phase == 2 {
    return LunaOnlineFramedServiceStreamActive
  }
  if self.stream_phase == 3 {
    ignore(self.progress_cooperatively())
    return self.readiness()
  }
  if !self.drain_requested {
    return LunaOnlineFramedServiceReady
  }
  let progress = if self.cooperative_handoff {
    self.cooperative_handoff = false
    self.progress_maintenance_core()
  } else {
    self.progress_reactor_core()
  }
  ignore(self.record_cooperative_progress(progress))
  self.readiness()
}
