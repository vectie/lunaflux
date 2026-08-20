///|
fn luna_event_view_stale() -> FramedWireError {
  StaleFrame
}

///|
fn LunaFramedEventAdapter::require_credit(
  self : LunaFramedEventAdapter,
) -> Unit raise FramedWireError {
  if self.busy {
    raise EventCreditUnavailable
  }
}

///|
fn LunaFramedEventAdapter::require_payload_capacity(
  self : LunaFramedEventAdapter,
  payload_length : Int,
) -> Int raise FramedWireError {
  if payload_length < 0 ||
    payload_length > self.storage.length() - EVENT_HEADER_BYTES {
    raise FrameTooLarge(frame=EventFrame)
  }
  EVENT_HEADER_BYTES + payload_length
}

///|
fn LunaFramedEventAdapter::publish(
  self : LunaFramedEventAdapter,
  length : Int,
) -> Unit {
  self.length = length
  self.busy = true
}

///|
fn LunaFramedEventAdapter::stage_accepted(
  self : LunaFramedEventAdapter,
  event : @luna_event.LunaEventView,
) -> Unit raise FramedWireError {
  let accepted = event.accepted() catch { _ => raise luna_event_view_stale() }
  let model_identity = accepted.model_identity() catch {
    _ => raise luna_event_view_stale()
  }
  let effective_limits = accepted.effective_limits() catch {
    _ => raise luna_event_view_stale()
  }
  let inference = self.limits.inference_limits()
  let max_new_tokens = effective_limits.max_new_tokens()
  let context_ceiling = effective_limits.context_ceiling()
  if max_new_tokens <= 0 ||
    max_new_tokens > inference.max_new_tokens() ||
    context_ceiling <= 0 ||
    context_ceiling > inference.max_context_tokens() ||
    max_new_tokens > context_ceiling {
    raise event_invalid(Scalar)
  }
  prepare_event_header(self.storage, 1, EVENT_HEADER_BYTES, 0, 0)
  write_i32(self.storage, 48, max_new_tokens)
  write_i32(self.storage, 52, context_ceiling)
  write_digest_ascii(self.storage, 64, model_identity.content().as_hex())
  write_digest_ascii(self.storage, 128, model_identity.plan().as_hex())
  write_u32(self.storage, 16, frame_checksum(self.storage, EVENT_HEADER_BYTES))
  self.publish(EVENT_HEADER_BYTES)
}

///|
fn LunaFramedEventAdapter::stage_token(
  self : LunaFramedEventAdapter,
  event : @luna_event.LunaEventView,
) -> Unit raise FramedWireError {
  let token = event.token() catch { _ => raise luna_event_view_stale() }
  let token_id = token.token_id() catch { _ => raise luna_event_view_stale() }
  let position = token.position() catch { _ => raise luna_event_view_stale() }
  let payload_length = token.delta_length() catch {
    _ => raise luna_event_view_stale()
  }
  let inference = self.limits.inference_limits()
  if token_id < 0 ||
    token_id > inference.max_token_id() ||
    position < 0 ||
    position >= inference.max_new_tokens() {
    raise event_invalid(Scalar)
  }
  if payload_length > inference.max_decoded_delta_bytes() {
    raise event_invalid(Length)
  }
  let total = self.require_payload_capacity(payload_length)
  prepare_event_header(
    self.storage,
    2,
    total,
    if payload_length == 0 {
      0
    } else {
      1
    },
    payload_length,
  )
  let copied = token.copy_delta_to(
    self.storage,
    destination_offset=EVENT_HEADER_BYTES,
  ) catch {
    _ => raise luna_event_view_stale()
  }
  if copied != payload_length {
    raise event_invalid(Payload)
  }
  write_i32(self.storage, 24, token_id)
  write_i32(self.storage, 28, position)
  write_u32(self.storage, 16, frame_checksum(self.storage, total))
  self.publish(total)
}

///|
fn LunaFramedEventAdapter::stage_usage(
  self : LunaFramedEventAdapter,
  event : @luna_event.LunaEventView,
) -> Unit raise FramedWireError {
  let usage = event.usage() catch { _ => raise luna_event_view_stale() }
  let input_tokens = usage.input_tokens() catch {
    _ => raise luna_event_view_stale()
  }
  let cached_input_tokens = usage.cached_input_tokens() catch {
    _ => raise luna_event_view_stale()
  }
  let output_tokens = usage.output_tokens() catch {
    _ => raise luna_event_view_stale()
  }
  if !valid_usage_scalars(
      self.limits.inference_limits(),
      input_tokens,
      cached_input_tokens,
      output_tokens,
    ) {
    raise event_invalid(Scalar)
  }
  prepare_event_header(self.storage, 3, EVENT_HEADER_BYTES, 0, 0)
  write_usage_scalars(
    self.storage,
    input_tokens,
    cached_input_tokens,
    output_tokens,
  )
  write_u32(self.storage, 16, frame_checksum(self.storage, EVENT_HEADER_BYTES))
  self.publish(EVENT_HEADER_BYTES)
}

///|
fn LunaFramedEventAdapter::stage_completed(
  self : LunaFramedEventAdapter,
  event : @luna_event.LunaEventView,
) -> Unit raise FramedWireError {
  let completed = event.completed() catch { _ => raise luna_event_view_stale() }
  let finish_reason = completed.finish_reason() catch {
    _ => raise luna_event_view_stale()
  }
  let input_tokens = completed.input_tokens() catch {
    _ => raise luna_event_view_stale()
  }
  let cached_input_tokens = completed.cached_input_tokens() catch {
    _ => raise luna_event_view_stale()
  }
  let output_tokens = completed.output_tokens() catch {
    _ => raise luna_event_view_stale()
  }
  let payload_length = completed.final_delta_length() catch {
    _ => raise luna_event_view_stale()
  }
  let inference = self.limits.inference_limits()
  if !valid_usage_scalars(
      inference, input_tokens, cached_input_tokens, output_tokens,
    ) {
    raise event_invalid(Scalar)
  }
  if payload_length > inference.max_decoded_delta_bytes() {
    raise event_invalid(Length)
  }
  let total = self.require_payload_capacity(payload_length)
  prepare_event_header(
    self.storage,
    4,
    total,
    if payload_length == 0 {
      0
    } else {
      1
    },
    payload_length,
  )
  write_i32(
    self.storage,
    24,
    match finish_reason {
      StopSequence => 0
      TokenLimit => 1
      Cancelled => 2
    },
  )
  write_usage_scalars(
    self.storage,
    input_tokens,
    cached_input_tokens,
    output_tokens,
  )
  let copied = completed.copy_final_delta_to(
    self.storage,
    destination_offset=EVENT_HEADER_BYTES,
  ) catch {
    _ => raise luna_event_view_stale()
  }
  if copied != payload_length {
    raise event_invalid(Payload)
  }
  write_u32(self.storage, 16, frame_checksum(self.storage, total))
  self.publish(total)
}

///|
fn LunaFramedEventAdapter::stage_failed(
  self : LunaFramedEventAdapter,
  event : @luna_event.LunaEventView,
) -> Unit raise FramedWireError {
  let failed = event.failed() catch { _ => raise luna_event_view_stale() }
  let retryable = failed.retryable() catch {
    _ => raise luna_event_view_stale()
  }
  let payload_length = failed.code_length() catch {
    _ => raise luna_event_view_stale()
  }
  let total = self.require_payload_capacity(payload_length)
  prepare_event_header(self.storage, 5, total, 0, payload_length)
  write_i32(self.storage, 24, if retryable { 1 } else { 0 })
  let copied = failed.copy_code_to(
    self.storage,
    destination_offset=EVENT_HEADER_BYTES,
  ) catch {
    _ => raise luna_event_view_stale()
  }
  if copied != payload_length {
    raise event_invalid(Payload)
  }
  write_u32(self.storage, 16, frame_checksum(self.storage, total))
  self.publish(total)
}

///|
pub fn LunaFramedEventAdapter::new(
  limits : FramedWireLimits,
) -> LunaFramedEventAdapter raise FramedWireError {
  let max_delta_bytes = limits.inference_limits().max_decoded_delta_bytes()
  let payload_envelope = if max_delta_bytes > 64 { max_delta_bytes } else { 64 }
  if limits.max_frame_bytes() < EVENT_HEADER_BYTES + payload_envelope {
    raise InvalidLimit(FrameBytes)
  }
  {
    limits,
    storage: FixedArray::make(limits.max_frame_bytes(), b'\x00'),
    busy: false,
    length: 0,
  }
}

///|
/// Stages one exact semantic event as canonical event-v2 bytes. This borrows
/// the view only; the semantic owner retains all retirement authority.
pub fn LunaFramedEventAdapter::stage(
  self : LunaFramedEventAdapter,
  event : @luna_event.LunaEventView,
) -> Unit raise FramedWireError {
  self.require_credit()
  let kind = event.kind() catch { _ => raise luna_event_view_stale() }
  match kind {
    LunaAcceptedEventKind => self.stage_accepted(event)
    LunaTokenEventKind => self.stage_token(event)
    LunaUsageEventKind => self.stage_usage(event)
    LunaCompletedEventKind => self.stage_completed(event)
    LunaFailedEventKind => self.stage_failed(event)
  }
}

///|
pub fn LunaFramedEventAdapter::length(
  self : LunaFramedEventAdapter,
) -> Int raise FramedWireError {
  if !self.busy {
    raise StaleFrame
  }
  self.length
}

///|
pub fn LunaFramedEventAdapter::copy_to(
  self : LunaFramedEventAdapter,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Int raise FramedWireError {
  let length = self.length()
  if destination_offset < 0 ||
    destination_offset > destination.length() - length {
    raise DestinationTooSmall
  }
  for index in 0..<length {
    destination[destination_offset + index] = self.storage[index]
  }
  length
}

///|
/// Releases only the adapter's transport credit. It never retires or
/// acknowledges the borrowed semantic event view.
pub fn LunaFramedEventAdapter::release(
  self : LunaFramedEventAdapter,
) -> Unit raise FramedWireError {
  if !self.busy {
    raise StaleFrame
  }
  self.length = 0
  self.busy = false
}
