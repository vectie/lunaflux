///|
fn valid_utf8_range(
  source : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> Bool {
  let end = offset + length
  let mut index = offset
  while index < end {
    let first = source[index].to_int()
    if first < 0x80 {
      index += 1
    } else if first >= 0xc2 && first <= 0xdf {
      if index + 1 >= end {
        return false
      }
      let second = source[index + 1].to_int()
      if second < 0x80 || second > 0xbf {
        return false
      }
      index += 2
    } else if first >= 0xe0 && first <= 0xef {
      if index + 2 >= end {
        return false
      }
      let second = source[index + 1].to_int()
      let third = source[index + 2].to_int()
      if third < 0x80 ||
        third > 0xbf ||
        second < 0x80 ||
        second > 0xbf ||
        (first == 0xe0 && second < 0xa0) ||
        (first == 0xed && second > 0x9f) {
        return false
      }
      index += 3
    } else if first >= 0xf0 && first <= 0xf4 {
      if index + 3 >= end {
        return false
      }
      let second = source[index + 1].to_int()
      let third = source[index + 2].to_int()
      let fourth = source[index + 3].to_int()
      if third < 0x80 ||
        third > 0xbf ||
        fourth < 0x80 ||
        fourth > 0xbf ||
        second < 0x80 ||
        second > 0xbf ||
        (first == 0xf0 && second < 0x90) ||
        (first == 0xf4 && second > 0x8f) {
        return false
      }
      index += 4
    } else {
      return false
    }
  }
  true
}

///|
fn prepare_event_header(
  bytes : FixedArray[Byte],
  kind : Int,
  length : Int,
  flags : UInt,
  payload_length : Int,
) -> Unit {
  clear_header(bytes, EVENT_HEADER_BYTES)
  write_u32(bytes, 0, event_magic)
  write_u32(bytes, 4, event_version)
  write_i32(bytes, 8, kind)
  write_i32(bytes, 12, length)
  write_u32(bytes, 20, flags)
  write_i32(bytes, 56, payload_length)
  write_i32(bytes, 196, EVENT_HEADER_BYTES)
}

///|
fn valid_public_code_range(
  source : FixedArray[Byte],
  offset : Int,
  length : Int,
) -> Bool {
  if length <= 0 ||
    length > 64 ||
    offset < 0 ||
    offset > source.length() - length {
    return false
  }
  for index in offset..<(offset + length) {
    let byte = source[index]
    if !((byte >= b'a' && byte <= b'z') ||
      (byte >= b'A' && byte <= b'Z') ||
      (byte >= b'0' && byte <= b'9') ||
      byte == b'-' ||
      byte == b'_' ||
      byte == b'.' ||
      byte == b':') {
      return false
    }
  }
  true
}

///|
fn write_digest_ascii(
  destination : FixedArray[Byte],
  offset : Int,
  value : String,
) -> Unit raise FramedWireError {
  if value.length() != 64 {
    raise event_invalid(Identity)
  }
  for index in 0..<64 {
    let unit = value[index]
    if !((unit >= ('0' : UInt16) && unit <= ('9' : UInt16)) ||
      (unit >= ('a' : UInt16) && unit <= ('f' : UInt16))) {
      raise event_invalid(Identity)
    }
    destination[offset + index] = unit.to_int().to_byte()
  }
}

///|
fn valid_usage_scalars(
  inference : @inference.InferenceLimits,
  input_tokens : Int,
  cached_input_tokens : Int,
  output_tokens : Int,
) -> Bool {
  input_tokens >= 0 &&
  input_tokens <= inference.max_context_tokens() &&
  cached_input_tokens >= 0 &&
  cached_input_tokens <= input_tokens &&
  output_tokens >= 0 &&
  output_tokens <= inference.max_new_tokens() &&
  input_tokens <= inference.max_context_tokens() - output_tokens
}

///|
fn write_usage_scalars(
  bytes : FixedArray[Byte],
  input_tokens : Int,
  cached_input_tokens : Int,
  output_tokens : Int,
) -> Unit {
  write_i32(bytes, 32, input_tokens)
  write_i32(bytes, 36, cached_input_tokens)
  write_i32(bytes, 40, output_tokens)
  write_i32(bytes, 44, input_tokens + output_tokens)
}

///|
fn CanonicalEventWriter::require_credit(
  self : CanonicalEventWriter,
) -> Unit raise FramedWireError {
  if self.busy {
    raise EventCreditUnavailable
  }
}

///|
fn CanonicalEventWriter::publish(
  self : CanonicalEventWriter,
  length : Int,
) -> Unit {
  copy_bytes(self.scratch, self.storage, length)
  self.length = length
  self.busy = true
}

///|
pub fn CanonicalEventWriter::new(
  limits : FramedWireLimits,
) -> CanonicalEventWriter {
  {
    limits,
    storage: FixedArray::make(limits.max_frame_bytes(), b'\x00'),
    scratch: FixedArray::make(limits.max_frame_bytes(), b'\x00'),
    busy: false,
    length: 0,
  }
}

///|
pub fn CanonicalEventWriter::capacity(self : CanonicalEventWriter) -> Int {
  self.storage.length()
}

///|
pub fn CanonicalEventWriter::has_credit(self : CanonicalEventWriter) -> Bool {
  !self.busy
}

///|
/// Preflights the canonical one-credit storage required by an online session:
/// the largest bounded UTF-8 delta/terminal tail and every public failure code.
/// This check is nonmutating and is intended to run before external authority
/// is acquired.
pub fn CanonicalEventWriter::preflight_online_capacity(
  self : CanonicalEventWriter,
  max_delta_bytes : Int,
) -> Unit raise FramedWireError {
  let payload_capacity = self.capacity() - EVENT_HEADER_BYTES
  if max_delta_bytes < 0 ||
    payload_capacity < max_delta_bytes ||
    payload_capacity < 64 {
    raise event_invalid(Length)
  }
}

///|
/// Canonically encodes a non-hot-path contract event. The prior frame remains
/// pinned until `retire` is called.
pub fn CanonicalEventWriter::prepare(
  self : CanonicalEventWriter,
  event : @inference.StreamEvent,
) -> Unit raise FramedWireError {
  self.require_credit()
  let length = encode_event_payload(
    self.scratch,
    event,
    self.limits.max_frame_bytes(),
  )
  ignore(parse_event(self.scratch, length, self.limits))
  self.publish(length)
}

///|
/// Writes canonical Accepted evidence without constructing a StreamEvent.
pub fn CanonicalEventWriter::prepare_accepted(
  self : CanonicalEventWriter,
  model_identity : @spec.ModelIdentity,
  effective_limits : @inference.EffectiveLimits,
) -> Unit raise FramedWireError {
  self.require_credit()
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
  prepare_event_header(self.scratch, 1, EVENT_HEADER_BYTES, 0, 0)
  write_i32(self.scratch, 48, max_new_tokens)
  write_i32(self.scratch, 52, context_ceiling)
  write_digest_ascii(self.scratch, 64, model_identity.content().as_hex())
  write_digest_ascii(self.scratch, 128, model_identity.plan().as_hex())
  write_u32(self.scratch, 16, frame_checksum(self.scratch, EVENT_HEADER_BYTES))
  self.publish(EVENT_HEADER_BYTES)
}

///|
/// Writes a canonical monotonic usage snapshot from bounded scalar evidence.
pub fn CanonicalEventWriter::prepare_usage(
  self : CanonicalEventWriter,
  input_tokens~ : Int,
  cached_input_tokens~ : Int,
  output_tokens~ : Int,
) -> Unit raise FramedWireError {
  self.require_credit()
  if !valid_usage_scalars(
      self.limits.inference_limits(),
      input_tokens,
      cached_input_tokens,
      output_tokens,
    ) {
    raise event_invalid(Scalar)
  }
  prepare_event_header(self.scratch, 3, EVENT_HEADER_BYTES, 0, 0)
  write_usage_scalars(
    self.scratch,
    input_tokens,
    cached_input_tokens,
    output_tokens,
  )
  write_u32(self.scratch, 16, frame_checksum(self.scratch, EVENT_HEADER_BYTES))
  self.publish(EVENT_HEADER_BYTES)
}

///|
/// Writes canonical Completed evidence and an optional final UTF-8 tail from
/// caller-owned fixed storage.
pub fn CanonicalEventWriter::prepare_completed(
  self : CanonicalEventWriter,
  finish_reason : @inference.FinishReason,
  input_tokens~ : Int,
  cached_input_tokens~ : Int,
  output_tokens~ : Int,
  tail : FixedArray[Byte],
  tail_offset~ : Int,
  tail_length~ : Int,
) -> Unit raise FramedWireError {
  self.require_credit()
  let inference = self.limits.inference_limits()
  if !valid_usage_scalars(
      inference, input_tokens, cached_input_tokens, output_tokens,
    ) {
    raise event_invalid(Scalar)
  }
  if tail_offset < 0 ||
    tail_length < 0 ||
    tail_offset > tail.length() ||
    tail_length > tail.length() - tail_offset ||
    tail_length > inference.max_decoded_delta_bytes() ||
    tail_length > self.limits.max_frame_bytes() - EVENT_HEADER_BYTES {
    raise event_invalid(Length)
  }
  if !valid_utf8_range(tail, tail_offset, tail_length) {
    raise event_invalid(Payload)
  }
  let total = EVENT_HEADER_BYTES + tail_length
  prepare_event_header(
    self.scratch,
    4,
    total,
    if tail_length == 0 {
      0
    } else {
      1
    },
    tail_length,
  )
  write_i32(
    self.scratch,
    24,
    match finish_reason {
      StopSequence => 0
      TokenLimit => 1
      Cancelled => 2
    },
  )
  write_usage_scalars(
    self.scratch,
    input_tokens,
    cached_input_tokens,
    output_tokens,
  )
  for index in 0..<tail_length {
    self.scratch[EVENT_HEADER_BYTES + index] = tail[tail_offset + index]
  }
  write_u32(self.scratch, 16, frame_checksum(self.scratch, total))
  self.publish(total)
}

///|
/// Writes canonical payload-safe Failed evidence from caller-owned ASCII.
pub fn CanonicalEventWriter::prepare_failed(
  self : CanonicalEventWriter,
  code : FixedArray[Byte],
  code_offset~ : Int,
  code_length~ : Int,
  retryable~ : Bool,
) -> Unit raise FramedWireError {
  self.require_credit()
  if !valid_public_code_range(code, code_offset, code_length) ||
    code_length > self.limits.max_frame_bytes() - EVENT_HEADER_BYTES {
    raise event_invalid(Payload)
  }
  let total = EVENT_HEADER_BYTES + code_length
  prepare_event_header(self.scratch, 5, total, 0, code_length)
  write_i32(self.scratch, 24, if retryable { 1 } else { 0 })
  for index in 0..<code_length {
    self.scratch[EVENT_HEADER_BYTES + index] = code[code_offset + index]
  }
  write_u32(self.scratch, 16, frame_checksum(self.scratch, total))
  self.publish(total)
}

///|
/// Writes the exact canonical Token event from caller-owned UTF-8 storage.
/// Validation and success perform no payload/string/event construction.
pub fn CanonicalEventWriter::prepare_token_bytes(
  self : CanonicalEventWriter,
  token_id~ : Int,
  position~ : Int,
  source : FixedArray[Byte],
  source_offset~ : Int,
  length~ : Int,
) -> Unit raise FramedWireError {
  self.require_credit()
  let inference = self.limits.inference_limits()
  if token_id < 0 ||
    token_id > inference.max_token_id() ||
    position < 0 ||
    position >= inference.max_new_tokens() {
    raise event_invalid(Scalar)
  }
  if source_offset < 0 ||
    length < 0 ||
    source_offset > source.length() ||
    length > source.length() - source_offset ||
    length > inference.max_decoded_delta_bytes() {
    raise event_invalid(Length)
  }
  if length > self.limits.max_frame_bytes() - EVENT_HEADER_BYTES {
    raise FrameTooLarge(frame=EventFrame)
  }
  if !valid_utf8_range(source, source_offset, length) {
    raise event_invalid(Payload)
  }
  let total = EVENT_HEADER_BYTES + length
  prepare_event_header(
    self.scratch,
    2,
    total,
    if length == 0 {
      0
    } else {
      1
    },
    length,
  )
  for index in 0..<length {
    self.scratch[EVENT_HEADER_BYTES + index] = source[source_offset + index]
  }
  write_i32(self.scratch, 24, token_id)
  write_i32(self.scratch, 28, position)
  write_u32(self.scratch, 16, frame_checksum(self.scratch, total))
  self.publish(total)
}

///|
pub fn CanonicalEventWriter::length(
  self : CanonicalEventWriter,
) -> Int raise FramedWireError {
  if !self.busy {
    raise StaleFrame
  }
  self.length
}

///|
pub fn CanonicalEventWriter::copy_to(
  self : CanonicalEventWriter,
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
pub fn CanonicalEventWriter::retire(
  self : CanonicalEventWriter,
) -> Unit raise FramedWireError {
  if !self.busy {
    raise StaleFrame
  }
  self.length = 0
  self.busy = false
}
