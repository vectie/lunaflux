///|
let event_magic : UInt = 0x5645464c

///|
let event_version : UInt = 1

///|
fn event_invalid(rule : FramedWireRule) -> FramedWireError {
  InvalidFrame(frame=EventFrame, rule~)
}

///|
fn event_kind(event : @inference.StreamEvent) -> Int {
  match event {
    Accepted(_) => 1
    Token(_) => 2
    Usage(_) => 3
    Completed(_) => 4
    Failed(_) => 5
  }
}

///|
fn write_usage(bytes : FixedArray[Byte], usage : @inference.Usage) -> Unit {
  write_i32(bytes, 32, usage.input_tokens())
  write_i32(bytes, 36, usage.cached_input_tokens())
  write_i32(bytes, 40, usage.output_tokens())
  write_i32(bytes, 44, usage.total_tokens())
}

///|
fn encode_event_payload(
  bytes : FixedArray[Byte],
  event : @inference.StreamEvent,
  maximum : Int,
) -> Int raise FramedWireError {
  clear_header(bytes, EVENT_HEADER_BYTES)
  let payload : Bytes = match event {
    Token(value) =>
      match value.decoded_delta() {
        Some(delta) => @utf8.encode(delta.text())
        None => b""
      }
    Failed(value) => @utf8.encode(value.code().as_string())
    _ => b""
  }
  if payload.length() > maximum - EVENT_HEADER_BYTES {
    raise FrameTooLarge(frame=EventFrame)
  }
  let length = EVENT_HEADER_BYTES + payload.length()
  clear_header(bytes, EVENT_HEADER_BYTES)
  ignore(write_bytes(bytes, EVENT_HEADER_BYTES, payload))
  write_u32(bytes, 0, event_magic)
  write_u32(bytes, 4, event_version)
  write_i32(bytes, 8, event_kind(event))
  write_i32(bytes, 12, length)
  write_i32(bytes, 56, payload.length())
  write_i32(bytes, 196, EVENT_HEADER_BYTES)
  match event {
    Accepted(value) => {
      write_i32(bytes, 48, value.effective_limits().max_new_tokens())
      write_i32(bytes, 52, value.effective_limits().context_ceiling())
      write_digest(
        bytes,
        64,
        value.model_identity().content().as_hex(),
        EventFrame,
      )
      write_digest(
        bytes,
        128,
        value.model_identity().plan().as_hex(),
        EventFrame,
      )
    }
    Token(value) => {
      write_i32(bytes, 24, value.token_id())
      write_i32(bytes, 28, value.position())
      match value.decoded_delta() {
        Some(_) => write_u32(bytes, 20, 1)
        None => ()
      }
    }
    Usage(value) => write_usage(bytes, value)
    Completed(value) => {
      write_i32(
        bytes,
        24,
        match value.finish_reason() {
          StopSequence => 0
          TokenLimit => 1
          Cancelled => 2
        },
      )
      write_usage(bytes, value.final_usage())
    }
    Failed(value) => write_i32(bytes, 24, if value.retryable() { 1 } else { 0 })
  }
  write_u32(bytes, 16, frame_checksum(bytes, length))
  length
}

///|
fn parse_usage(
  bytes : FixedArray[Byte],
  limits : @inference.InferenceLimits,
) -> @inference.Usage raise FramedWireError {
  let usage = @inference.Usage::initial(
    input_tokens=read_i32(bytes, 32),
    cached_input_tokens=read_i32(bytes, 36),
    limits,
  ) catch {
    _ => raise event_invalid(Scalar)
  }
  let advanced = usage.advance(output_tokens=read_i32(bytes, 40), limits) catch {
    _ => raise event_invalid(Scalar)
  }
  if advanced.total_tokens() != read_i32(bytes, 44) {
    raise event_invalid(CanonicalEncoding)
  }
  advanced
}

///|
fn require_event_zero(
  bytes : FixedArray[Byte],
  start : Int,
  end : Int,
) -> Unit raise FramedWireError {
  for index in start..<end {
    if bytes[index] != b'\x00' {
      raise event_invalid(Reserved)
    }
  }
}

///|
fn parse_event(
  bytes : FixedArray[Byte],
  length : Int,
  limits : FramedWireLimits,
) -> @inference.StreamEvent raise FramedWireError {
  if length < EVENT_HEADER_BYTES ||
    length > limits.max_frame_bytes() ||
    length > bytes.length() {
    raise event_invalid(Length)
  }
  if read_u32(bytes, 0) != event_magic {
    raise event_invalid(Magic)
  }
  if read_u32(bytes, 4) != event_version {
    raise event_invalid(Version)
  }
  let kind = read_i32(bytes, 8)
  if kind < 1 || kind > 5 {
    raise event_invalid(Kind)
  }
  if read_i32(bytes, 12) != length ||
    read_i32(bytes, 56) != length - EVENT_HEADER_BYTES {
    raise event_invalid(Length)
  }
  if read_u32(bytes, 16) != frame_checksum(bytes, length) {
    raise event_invalid(Checksum)
  }
  let flags = read_u32(bytes, 20)
  if (flags & 0xfffffffe) != 0 || read_i32(bytes, 196) != EVENT_HEADER_BYTES {
    raise event_invalid(Flags)
  }
  require_event_zero(bytes, 60, 64)
  require_event_zero(bytes, 192, 196)
  require_event_zero(bytes, 200, 208)
  let inference = limits.inference_limits()
  match kind {
    1 => {
      if flags != 0 || length != EVENT_HEADER_BYTES {
        raise event_invalid(CanonicalEncoding)
      }
      require_event_zero(bytes, 24, 48)
      let content = @spec.ContentDigest::from_sha256(
        read_digest(bytes, 64, EventFrame),
      ) catch {
        _ => raise event_invalid(Identity)
      }
      let plan = @spec.PlanDigest::from_sha256(
        read_digest(bytes, 128, EventFrame),
      ) catch {
        _ => raise event_invalid(Identity)
      }
      let effective = @inference.EffectiveLimits::new(
        max_new_tokens=read_i32(bytes, 48),
        context_ceiling=read_i32(bytes, 52),
        inference,
      ) catch {
        _ => raise event_invalid(Scalar)
      }
      Accepted(
        @inference.AcceptedEvent::new(
          model_identity=@spec.ModelIdentity::new(content, plan),
          effective_limits=effective,
        ),
      )
    }
    2 => {
      require_event_zero(bytes, 32, 56)
      require_event_zero(bytes, 64, 192)
      let payload_length = length - EVENT_HEADER_BYTES
      if payload_length > inference.max_decoded_delta_bytes() {
        raise event_invalid(Length)
      }
      let payload = read_bytes(bytes, EVENT_HEADER_BYTES, payload_length)
      if ((flags & 1) != 0) != (payload.length() > 0) {
        raise event_invalid(CanonicalEncoding)
      }
      let delta = if (flags & 1) != 0 {
        Some(
          @inference.DecodedDelta::from_utf8(payload, inference) catch {
            _ => raise event_invalid(Payload)
          },
        )
      } else {
        None
      }
      Token(
        @inference.TokenEvent::new(
          token_id=read_i32(bytes, 24),
          decoded_delta=delta,
          position=read_i32(bytes, 28),
          inference,
        ) catch {
          _ => raise event_invalid(Scalar)
        },
      )
    }
    3 => {
      if flags != 0 || length != EVENT_HEADER_BYTES {
        raise event_invalid(CanonicalEncoding)
      }
      require_event_zero(bytes, 24, 32)
      require_event_zero(bytes, 48, 56)
      require_event_zero(bytes, 64, 192)
      Usage(parse_usage(bytes, inference))
    }
    4 => {
      if flags != 0 || length != EVENT_HEADER_BYTES {
        raise event_invalid(CanonicalEncoding)
      }
      require_event_zero(bytes, 28, 32)
      require_event_zero(bytes, 48, 56)
      require_event_zero(bytes, 64, 192)
      let reason = match read_i32(bytes, 24) {
        0 => @inference.StopSequence
        1 => TokenLimit
        2 => Cancelled
        _ => raise event_invalid(Scalar)
      }
      Completed(
        @inference.CompletedEvent::new(
          finish_reason=reason,
          final_usage=parse_usage(bytes, inference),
        ),
      )
    }
    5 => {
      if flags != 0 {
        raise event_invalid(CanonicalEncoding)
      }
      require_event_zero(bytes, 28, 56)
      require_event_zero(bytes, 64, 192)
      let retryable = match read_i32(bytes, 24) {
        0 => false
        1 => true
        _ => raise event_invalid(Scalar)
      }
      let payload_length = length - EVENT_HEADER_BYTES
      if payload_length == 0 || payload_length > 64 {
        raise event_invalid(Length)
      }
      let code = @inference.PublicErrorCode::from_ascii(
        read_bytes(bytes, EVENT_HEADER_BYTES, payload_length),
      ) catch {
        _ => raise event_invalid(Payload)
      }
      Failed(@inference.FailedEvent::new(code~, retryable~))
    }
    _ => raise event_invalid(Kind)
  }
}

///|
pub fn EventFrameBuffer::encode(
  self : EventFrameBuffer,
  event : @inference.StreamEvent,
) -> ValidatedEventFrame raise FramedWireError {
  let length = encode_event_payload(
    self.scratch,
    event,
    self.limits.max_frame_bytes(),
  )
  let canonical = parse_event(self.scratch, length, self.limits)
  self.publish(canonical, length)
}

///|
/// Validates before replacing the current frame; rejection is transactional.
pub fn EventFrameBuffer::load(
  self : EventFrameBuffer,
  source : FixedArray[Byte],
  length : Int,
) -> ValidatedEventFrame raise FramedWireError {
  let event = parse_event(source, length, self.limits)
  copy_bytes(source, self.scratch, length)
  self.publish(event, length)
}
