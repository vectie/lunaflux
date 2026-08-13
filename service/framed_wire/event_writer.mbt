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
