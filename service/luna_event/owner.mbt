///|
/// Allocates one persistent semantic event slot. Its payload storage covers
/// both the configured decoded-delta envelope and the fixed 64-byte public
/// error-code envelope.
pub fn LunaEventOwner::new(
  limits : @inference.InferenceLimits,
) -> LunaEventOwner {
  let payload_capacity = if limits.max_decoded_delta_bytes() > 64 {
    limits.max_decoded_delta_bytes()
  } else {
    64
  }
  {
    limits,
    payload: FixedArray::make(payload_capacity, b'\x00'),
    model_identity: None,
    effective_limits: None,
    token_id: 0,
    position: 0,
    input_tokens: 0,
    cached_input_tokens: 0,
    output_tokens: 0,
    finish_reason: StopSequence,
    retryable: false,
    payload_length: 0,
    epoch: 0UL,
    kind: LunaNoEventSlot,
    busy: false,
  }
}

///|
pub fn LunaEventOwner::has_event(self : LunaEventOwner) -> Bool {
  self.busy
}

///|
/// Reports whether exactly `event_count` future publications can advance the
/// persistent epoch without wrapping. This read-only preflight neither
/// reserves credit nor changes the current event.
pub fn LunaEventOwner::has_epoch_capacity(
  self : LunaEventOwner,
  event_count : Int,
) -> Bool {
  event_count >= 0 &&
  self.epoch <= 0xffffffffffffffffUL - event_count.to_uint64()
}

///|
fn LunaEventOwner::commit(
  self : LunaEventOwner,
  kind : LunaEventSlotKind,
  payload_length : Int,
) -> Unit {
  self.payload_length = payload_length
  self.epoch += 1UL
  self.kind = kind
  self.busy = true
}

///|
fn LunaEventOwner::require_publishable(
  self : LunaEventOwner,
) -> Unit raise LunaEventError {
  if self.busy {
    raise luna_event_invalid(LunaEventLifecycle)
  }
  if self.epoch == 0xffffffffffffffffUL {
    raise luna_event_invalid(LunaEventEpoch)
  }
}

///|
/// Publishes typed Accepted evidence without constructing a StreamEvent.
pub fn LunaEventOwner::publish_accepted(
  self : LunaEventOwner,
  model_identity : @spec.ModelIdentity,
  effective_limits : @inference.EffectiveLimits,
) -> Unit raise LunaEventError {
  self.require_publishable()
  let max_new_tokens = effective_limits.max_new_tokens()
  let context_ceiling = effective_limits.context_ceiling()
  if max_new_tokens <= 0 ||
    max_new_tokens > self.limits.max_new_tokens() ||
    context_ceiling <= 0 ||
    context_ceiling > self.limits.max_context_tokens() ||
    max_new_tokens > context_ceiling {
    raise luna_event_invalid(LunaEventScalar)
  }
  self.model_identity = Some(model_identity)
  self.effective_limits = Some(effective_limits)
  self.commit(LunaAcceptedEventSlot, 0)
}

///|
/// Attempts to publish a Token from caller-owned fixed storage. Rejection is
/// allocation-free and leaves any prior semantic credit unchanged.
pub fn LunaEventOwner::publish_token_status(
  self : LunaEventOwner,
  token_id~ : Int,
  position~ : Int,
  source : FixedArray[Byte],
  source_offset~ : Int,
  length~ : Int,
) -> LunaEventPublishStatus {
  if self.busy {
    return LunaEventPublishCreditUnavailable
  }
  if self.epoch == 0xffffffffffffffffUL {
    return LunaEventPublishEpochExhausted
  }
  if token_id < 0 ||
    token_id > self.limits.max_token_id() ||
    position < 0 ||
    position >= self.limits.max_new_tokens() ||
    !luna_valid_source_range(
      source,
      source_offset,
      length,
      self.limits.max_decoded_delta_bytes(),
    ) ||
    !luna_valid_utf8_range(source, source_offset, length) {
    return LunaEventPublishPayloadInvalid
  }
  for index in 0..<length {
    self.payload[index] = source[source_offset + index]
  }
  self.token_id = token_id
  self.position = position
  self.commit(LunaTokenEventSlot, length)
  LunaEventPublishReady
}

///|
pub fn LunaEventOwner::publish_usage(
  self : LunaEventOwner,
  input_tokens~ : Int,
  cached_input_tokens~ : Int,
  output_tokens~ : Int,
) -> Unit raise LunaEventError {
  self.require_publishable()
  if !luna_valid_usage(
      self.limits,
      input_tokens,
      cached_input_tokens,
      output_tokens,
    ) {
    raise luna_event_invalid(LunaEventScalar)
  }
  self.input_tokens = input_tokens
  self.cached_input_tokens = cached_input_tokens
  self.output_tokens = output_tokens
  self.commit(LunaUsageEventSlot, 0)
}

///|
pub fn LunaEventOwner::publish_completed(
  self : LunaEventOwner,
  finish_reason : @inference.FinishReason,
  input_tokens~ : Int,
  cached_input_tokens~ : Int,
  output_tokens~ : Int,
  tail : FixedArray[Byte],
  tail_offset~ : Int,
  tail_length~ : Int,
) -> Unit raise LunaEventError {
  self.require_publishable()
  if !luna_valid_usage(
      self.limits,
      input_tokens,
      cached_input_tokens,
      output_tokens,
    ) {
    raise luna_event_invalid(LunaEventScalar)
  }
  if !luna_valid_source_range(
      tail,
      tail_offset,
      tail_length,
      self.limits.max_decoded_delta_bytes(),
    ) {
    raise luna_event_invalid(LunaEventLength)
  }
  if !luna_valid_utf8_range(tail, tail_offset, tail_length) {
    raise luna_event_invalid(LunaEventPayload)
  }
  for index in 0..<tail_length {
    self.payload[index] = tail[tail_offset + index]
  }
  self.finish_reason = finish_reason
  self.input_tokens = input_tokens
  self.cached_input_tokens = cached_input_tokens
  self.output_tokens = output_tokens
  self.commit(LunaCompletedEventSlot, tail_length)
}

///|
pub fn LunaEventOwner::publish_failed(
  self : LunaEventOwner,
  code : FixedArray[Byte],
  code_offset~ : Int,
  code_length~ : Int,
  retryable~ : Bool,
) -> Unit raise LunaEventError {
  self.require_publishable()
  if !luna_valid_public_code_range(code, code_offset, code_length) {
    raise luna_event_invalid(LunaEventPayload)
  }
  for index in 0..<code_length {
    self.payload[index] = code[code_offset + index]
  }
  self.retryable = retryable
  self.commit(LunaFailedEventSlot, code_length)
}

///|
fn LunaEventOwner::require_exact_usage(
  self : LunaEventOwner,
  view : LunaUsageEventView,
) -> Unit raise LunaEventError {
  if !self.busy ||
    !physical_equal(view.owner, self) ||
    view.epoch != self.epoch ||
    self.kind != LunaUsageEventSlot {
    raise luna_event_invalid(LunaEventEpoch)
  }
  if self.epoch == 0xffffffffffffffffUL {
    raise luna_event_invalid(LunaEventEpoch)
  }
}

///|
/// Atomically replaces the exact pinned Usage epoch with Completed. All view,
/// scalar, range, UTF-8, and next-epoch checks precede payload or owner
/// mutation; success publishes a distinct, non-wrapping event epoch.
pub fn LunaEventOwner::replace_usage_with_completed(
  self : LunaEventOwner,
  usage : LunaUsageEventView,
  finish_reason : @inference.FinishReason,
  tail : FixedArray[Byte],
  tail_offset~ : Int,
  tail_length~ : Int,
) -> LunaCompletedEventView raise LunaEventError {
  self.require_exact_usage(usage)
  if !luna_valid_source_range(
      tail,
      tail_offset,
      tail_length,
      self.limits.max_decoded_delta_bytes(),
    ) {
    raise luna_event_invalid(LunaEventLength)
  }
  if !luna_valid_utf8_range(tail, tail_offset, tail_length) {
    raise luna_event_invalid(LunaEventPayload)
  }
  for index in 0..<tail_length {
    self.payload[index] = tail[tail_offset + index]
  }
  self.finish_reason = finish_reason
  self.epoch += 1UL
  self.kind = LunaCompletedEventSlot
  self.payload_length = tail_length
  { owner: self, epoch: self.epoch }
}

///|
/// Atomically replaces the exact pinned Usage epoch with Failed. Rejection
/// leaves the Usage view and its scalar evidence unchanged.
pub fn LunaEventOwner::replace_usage_with_failed(
  self : LunaEventOwner,
  usage : LunaUsageEventView,
  code : FixedArray[Byte],
  code_offset~ : Int,
  code_length~ : Int,
  retryable~ : Bool,
) -> LunaFailedEventView raise LunaEventError {
  self.require_exact_usage(usage)
  if !luna_valid_public_code_range(code, code_offset, code_length) {
    raise luna_event_invalid(LunaEventPayload)
  }
  for index in 0..<code_length {
    self.payload[index] = code[code_offset + index]
  }
  self.retryable = retryable
  self.epoch += 1UL
  self.kind = LunaFailedEventSlot
  self.payload_length = code_length
  { owner: self, epoch: self.epoch }
}

///|
/// Borrows an immutable typed view of the exact pinned event epoch.
pub fn LunaEventOwner::view(
  self : LunaEventOwner,
) -> LunaEventView raise LunaEventError {
  if !self.busy {
    raise luna_event_invalid(LunaEventLifecycle)
  }
  { owner: self, epoch: self.epoch }
}

///|
fn LunaEventOwner::matches_view(
  self : LunaEventOwner,
  view : LunaEventView,
) -> Bool {
  physical_equal(view.owner, self) && view.epoch == self.epoch
}

///|
fn LunaEventOwner::clear_event(self : LunaEventOwner) -> Unit {
  self.model_identity = None
  self.effective_limits = None
  self.payload_length = 0
  self.kind = LunaNoEventSlot
  self.busy = false
}

///|
/// Retires only the exact view borrowed from this owner and event epoch.
pub fn LunaEventOwner::retire(
  self : LunaEventOwner,
  view : LunaEventView,
) -> Unit raise LunaEventError {
  if !self.busy || !self.matches_view(view) {
    raise luna_event_invalid(LunaEventEpoch)
  }
  self.clear_event()
}

///|
/// Invalidates the currently pinned event without granting its consumer an
/// acknowledgement. This is the disconnect/abort path.
pub fn LunaEventOwner::discard(
  self : LunaEventOwner,
) -> Unit raise LunaEventError {
  if !self.busy {
    raise luna_event_invalid(LunaEventLifecycle)
  }
  self.clear_event()
}
