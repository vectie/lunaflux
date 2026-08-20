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
