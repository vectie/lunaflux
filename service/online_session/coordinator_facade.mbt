///|
pub fn LunaOnlineFramedCoordinator::offer_luna_framed(
  self : LunaOnlineFramedCoordinator,
  source : FixedArray[Byte],
  source_offset~ : Int,
  length~ : Int,
) -> LunaOnlineFramedIngress raise LunaOnlineFramedCoordinatorError {
  self.stream.offer_luna_framed(source, source_offset~, length~)
}

///|
pub fn LunaOnlineFramedCoordinator::transport_wait_remaining_millis(
  self : LunaOnlineFramedCoordinator,
) -> Int raise LunaOnlineFramedCoordinatorError {
  self.stream.transport_wait_remaining_millis()
}

///|
pub fn LunaOnlineFramedCoordinator::maintenance_wait_remaining_millis(
  self : LunaOnlineFramedCoordinator,
) -> Int raise LunaOnlineFramedCoordinatorError {
  self.stream.maintenance_wait_remaining_millis()
}

///|
pub fn LunaOnlineFramedCoordinator::request_cancel(
  self : LunaOnlineFramedCoordinator,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  self.stream.request_cancel()
}

///|
pub fn LunaOnlineFramedCoordinator::disconnect(
  self : LunaOnlineFramedCoordinator,
) -> Unit raise LunaOnlineFramedCoordinatorError {
  if !self.service.closed && self.service.stream_phase != 0 {
    self.stream.disconnect()
  }
  self.service.begin_drain()
}

///|
pub fn LunaOnlineFramedCoordinator::begin_drain(
  self : LunaOnlineFramedCoordinator,
) -> Unit {
  self.service.begin_drain()
}

///|
fn LunaOnlineFramedCoordinator::consume_compatibility_observation(
  self : LunaOnlineFramedCoordinator,
  progress : LunaOnlineFramedCoordinatorProgress,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  if progress != LunaOnlineFramedCoordinatorObservationReady {
    return progress
  }
  let observation = self.stream.take_observation()
  observation.ack()
  LunaOnlineFramedCoordinatorAdvanced
}

///|
pub fn LunaOnlineFramedCoordinator::progress(
  self : LunaOnlineFramedCoordinator,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  if self.service.closed {
    return LunaOnlineFramedCoordinatorClosed
  }
  if self.service.disconnect_requested && !self.service.drain_requested {
    self.service.begin_drain()
  }
  let progress = self.consume_compatibility_observation(
    self.service.progress_reactor_core(),
  )
  if self.service.disconnect_requested && !self.service.drain_requested {
    self.service.begin_drain()
  }
  if self.service.stream_phase == 3 &&
    self.service.finish_stream_retirement_if_ready() {
    return progress
  }
  progress
}

///|
pub fn LunaOnlineFramedCoordinator::progress_off_reactor_maintenance(
  self : LunaOnlineFramedCoordinator,
) -> LunaOnlineFramedCoordinatorProgress raise LunaOnlineFramedCoordinatorError {
  if self.service.stream_phase == 3 &&
    self.service.finish_stream_retirement_if_ready() {
    return self.consume_compatibility_observation(
      self.service.progress_maintenance_core(),
    )
  }
  self.consume_compatibility_observation(
    self.service.progress_maintenance_core(),
  )
}

///|
pub fn LunaOnlineFramedCoordinator::framed_event_remaining(
  self : LunaOnlineFramedCoordinator,
) -> Int raise LunaOnlineFramedCoordinatorError {
  self.stream.framed_event_remaining()
}

///|
pub fn LunaOnlineFramedCoordinator::copy_framed_event_chunk_to(
  self : LunaOnlineFramedCoordinator,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
  maximum_length~ : Int,
) -> LunaOnlineFramedEventOffer raise LunaOnlineFramedCoordinatorError {
  self.stream.copy_framed_event_chunk_to(
    destination,
    destination_offset~,
    maximum_length~,
  )
}

///|
pub fn LunaOnlineFramedCoordinator::take_rejection(
  self : LunaOnlineFramedCoordinator,
) -> LunaOnlineFramedRejectionCredit raise LunaOnlineFramedCoordinatorError {
  self.stream.take_rejection()
}

///|
pub fn LunaOnlineFramedCoordinator::is_closed(
  self : LunaOnlineFramedCoordinator,
) -> Bool {
  self.service.is_closed()
}
