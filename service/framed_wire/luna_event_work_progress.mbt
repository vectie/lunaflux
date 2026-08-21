///|
fn luna_event_prepare_header_fields(
  bytes : FixedArray[Byte],
  kind : Int,
  length : Int,
  flags : UInt,
  payload_length : Int,
) -> Unit {
  write_u32(bytes, 0, event_magic)
  write_u32(bytes, 4, event_version)
  write_i32(bytes, 8, kind)
  write_i32(bytes, 12, length)
  write_u32(bytes, 20, flags)
  write_i32(bytes, 56, payload_length)
  write_i32(bytes, 196, EVENT_HEADER_BYTES)
}

///|
fn LunaFramedEventWorkspace::require_event_capacity(
  self : LunaFramedEventWorkspace,
  payload_length : Int,
) -> Int raise FramedWireError {
  if payload_length < 0 ||
    payload_length > self.storage.length() - EVENT_HEADER_BYTES {
    raise FrameTooLarge(frame=EventFrame)
  }
  EVENT_HEADER_BYTES + payload_length
}

///|
fn LunaFramedEventWorkspace::setup_accepted(
  self : LunaFramedEventWorkspace,
  event : @luna_event.LunaEventView,
) -> Unit raise FramedWireError {
  let accepted = event.accepted() catch { _ => raise StaleFrame }
  let effective = accepted.effective_limits() catch { _ => raise StaleFrame }
  let max_new_tokens = effective.max_new_tokens()
  let context_ceiling = effective.context_ceiling()
  let inference = self.limits.inference_limits()
  if max_new_tokens <= 0 ||
    max_new_tokens > inference.max_new_tokens() ||
    context_ceiling <= 0 ||
    context_ceiling > inference.max_context_tokens() ||
    max_new_tokens > context_ceiling {
    raise event_invalid(Scalar)
  }
  self.length = EVENT_HEADER_BYTES
  self.payload_length = 128
  luna_event_prepare_header_fields(self.storage, 1, self.length, 0, 0)
  write_i32(self.storage, 48, max_new_tokens)
  write_i32(self.storage, 52, context_ceiling)
}

///|
fn LunaFramedEventWorkspace::setup_token(
  self : LunaFramedEventWorkspace,
  event : @luna_event.LunaEventView,
) -> Unit raise FramedWireError {
  let token = event.token() catch { _ => raise StaleFrame }
  let token_id = token.token_id() catch { _ => raise StaleFrame }
  let position = token.position() catch { _ => raise StaleFrame }
  let payload = token.delta_length() catch { _ => raise StaleFrame }
  let inference = self.limits.inference_limits()
  if token_id < 0 ||
    token_id > inference.max_token_id() ||
    position < 0 ||
    position >= inference.max_new_tokens() {
    raise event_invalid(Scalar)
  }
  if payload > inference.max_decoded_delta_bytes() {
    raise event_invalid(Length)
  }
  self.length = self.require_event_capacity(payload)
  self.payload_length = payload
  luna_event_prepare_header_fields(
    self.storage,
    2,
    self.length,
    if payload == 0 {
      0
    } else {
      1
    },
    payload,
  )
  write_i32(self.storage, 24, token_id)
  write_i32(self.storage, 28, position)
}

///|
fn LunaFramedEventWorkspace::setup_usage(
  self : LunaFramedEventWorkspace,
  event : @luna_event.LunaEventView,
) -> Unit raise FramedWireError {
  let usage = event.usage() catch { _ => raise StaleFrame }
  let input = usage.input_tokens() catch { _ => raise StaleFrame }
  let cached = usage.cached_input_tokens() catch { _ => raise StaleFrame }
  let output = usage.output_tokens() catch { _ => raise StaleFrame }
  if !valid_usage_scalars(self.limits.inference_limits(), input, cached, output) {
    raise event_invalid(Scalar)
  }
  self.length = EVENT_HEADER_BYTES
  self.payload_length = 0
  luna_event_prepare_header_fields(self.storage, 3, self.length, 0, 0)
  write_usage_scalars(self.storage, input, cached, output)
}

///|
fn LunaFramedEventWorkspace::setup_completed(
  self : LunaFramedEventWorkspace,
  event : @luna_event.LunaEventView,
) -> Unit raise FramedWireError {
  let completed = event.completed() catch { _ => raise StaleFrame }
  let reason = completed.finish_reason() catch { _ => raise StaleFrame }
  let input = completed.input_tokens() catch { _ => raise StaleFrame }
  let cached = completed.cached_input_tokens() catch { _ => raise StaleFrame }
  let output = completed.output_tokens() catch { _ => raise StaleFrame }
  let payload = completed.final_delta_length() catch { _ => raise StaleFrame }
  let inference = self.limits.inference_limits()
  if !valid_usage_scalars(inference, input, cached, output) {
    raise event_invalid(Scalar)
  }
  if payload > inference.max_decoded_delta_bytes() {
    raise event_invalid(Length)
  }
  self.length = self.require_event_capacity(payload)
  self.payload_length = payload
  luna_event_prepare_header_fields(
    self.storage,
    4,
    self.length,
    if payload == 0 {
      0
    } else {
      1
    },
    payload,
  )
  write_i32(
    self.storage,
    24,
    match reason {
      StopSequence => 0
      TokenLimit => 1
      Cancelled => 2
    },
  )
  write_usage_scalars(self.storage, input, cached, output)
}

///|
fn LunaFramedEventWorkspace::setup_failed(
  self : LunaFramedEventWorkspace,
  event : @luna_event.LunaEventView,
) -> Unit raise FramedWireError {
  let failed = event.failed() catch { _ => raise StaleFrame }
  let retryable = failed.retryable() catch { _ => raise StaleFrame }
  let payload = failed.code_length() catch { _ => raise StaleFrame }
  self.length = self.require_event_capacity(payload)
  self.payload_length = payload
  luna_event_prepare_header_fields(self.storage, 5, self.length, 0, payload)
  write_i32(self.storage, 24, if retryable { 1 } else { 0 })
}

///|
fn LunaFramedEventWorkspace::progress_setup(
  self : LunaFramedEventWorkspace,
  event : @luna_event.LunaEventView,
) -> Unit raise FramedWireError {
  match self.event_kind {
    1 => self.setup_accepted(event)
    2 => self.setup_token(event)
    3 => self.setup_usage(event)
    4 => self.setup_completed(event)
    5 => self.setup_failed(event)
    _ => raise event_invalid(Kind)
  }
  self.cursor = 0
  self.phase = if self.payload_length == 0 {
    LUNA_FRAMED_EVENT_CHECKSUM
  } else {
    LUNA_FRAMED_EVENT_COPYING
  }
}

///|
fn valid_digest_ascii(byte : Byte) -> Bool {
  (byte >= b'0' && byte <= b'9') || (byte >= b'a' && byte <= b'f')
}

///|
fn LunaFramedEventWorkspace::progress_copy(
  self : LunaFramedEventWorkspace,
  event : @luna_event.LunaEventView,
) -> Unit raise FramedWireError {
  let byte = match self.event_kind {
    1 => {
      let accepted = event.accepted() catch { _ => raise StaleFrame }
      let value = if self.cursor < 64 {
        accepted.content_digest_byte_at(self.cursor) catch {
          _ => raise StaleFrame
        }
      } else {
        accepted.plan_digest_byte_at(self.cursor - 64) catch {
          _ => raise StaleFrame
        }
      }
      if !valid_digest_ascii(value) {
        raise event_invalid(Identity)
      }
      value
    }
    2 => {
      let token = event.token() catch { _ => raise StaleFrame }
      token.delta_byte_at(self.cursor) catch {
        _ => raise StaleFrame
      }
    }
    4 => {
      let completed = event.completed() catch { _ => raise StaleFrame }
      completed.final_delta_byte_at(self.cursor) catch {
        _ => raise StaleFrame
      }
    }
    5 => {
      let failed = event.failed() catch { _ => raise StaleFrame }
      failed.code_byte_at(self.cursor) catch {
        _ => raise StaleFrame
      }
    }
    _ => abort("Luna framed event copy-kind invariant")
  }
  let destination = if self.event_kind == 1 {
    if self.cursor < 64 {
      64 + self.cursor
    } else {
      128 + self.cursor - 64
    }
  } else {
    EVENT_HEADER_BYTES + self.cursor
  }
  self.storage[destination] = byte
  self.cursor += 1
  if self.cursor == self.payload_length {
    self.cursor = 0
    self.phase = LUNA_FRAMED_EVENT_CHECKSUM
  }
}

///|
fn LunaFramedEventWorkspace::progress_checksum(
  self : LunaFramedEventWorkspace,
) -> Unit {
  if self.cursor < 16 || self.cursor >= 20 {
    self.checksum = (self.checksum ^ self.storage[self.cursor].to_uint()) *
      16777619
  }
  self.cursor += 1
  if self.cursor == self.length {
    self.phase = LUNA_FRAMED_EVENT_FINALIZING
  }
}

///|
fn LunaFramedEventWorkspace::progress_one(
  self : LunaFramedEventWorkspace,
) -> Unit raise FramedWireError {
  let event = self.authenticate_event()
  match self.phase {
    LUNA_FRAMED_EVENT_CLEARING => {
      self.storage[self.cursor] = b'\x00'
      self.cursor += 1
      if self.cursor == EVENT_HEADER_BYTES {
        self.cursor = 0
        self.phase = LUNA_FRAMED_EVENT_SETUP
      }
    }
    LUNA_FRAMED_EVENT_SETUP => self.progress_setup(event)
    LUNA_FRAMED_EVENT_COPYING => self.progress_copy(event)
    LUNA_FRAMED_EVENT_CHECKSUM => self.progress_checksum()
    LUNA_FRAMED_EVENT_FINALIZING => {
      // The authentication above is the final semantic liveness proof. The
      // canonical frame is immutable and independent after this transition.
      write_u32(self.storage, 16, self.checksum)
      self.semantic_views.clear()
      self.phase = LUNA_FRAMED_EVENT_READY
    }
    _ => abort("Luna framed event progress-phase invariant")
  }
}

///|
pub fn LunaFramedEventWork::progress(
  self : LunaFramedEventWork,
) -> LunaFramedEventProgress raise FramedWireError {
  let workspace = self.require_workspace()
  workspace.last_work_units = 0
  if workspace.phase == LUNA_FRAMED_EVENT_READY ||
    workspace.phase == LUNA_FRAMED_EVENT_FAILED {
    return LunaFramedEventIdle
  }
  let budget = workspace.budget.work_units()
  let mut used = 0
  while used < budget &&
        workspace.phase != LUNA_FRAMED_EVENT_READY &&
        workspace.phase != LUNA_FRAMED_EVENT_FAILED {
    workspace.progress_one() catch {
      error => workspace.fail(error)
    }
    used += 1
  }
  workspace.last_work_units = used
  workspace.total_work_units += used.to_uint64()
  LunaFramedEventAdvanced
}
