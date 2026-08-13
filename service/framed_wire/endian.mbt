///|
fn write_u32(bytes : FixedArray[Byte], offset : Int, value : UInt) -> Unit {
  bytes.unsafe_write_uint32_le(offset, value)
}

///|
fn write_i32(bytes : FixedArray[Byte], offset : Int, value : Int) -> Unit {
  write_u32(bytes, offset, value.reinterpret_as_uint())
}

///|
fn write_u64(bytes : FixedArray[Byte], offset : Int, value : UInt64) -> Unit {
  bytes.unsafe_write_uint64_le(offset, value)
}

///|
fn read_u32(bytes : FixedArray[Byte], offset : Int) -> UInt {
  bytes[offset].to_uint() |
  (bytes[offset + 1].to_uint() << 8) |
  (bytes[offset + 2].to_uint() << 16) |
  (bytes[offset + 3].to_uint() << 24)
}

///|
fn read_i32(bytes : FixedArray[Byte], offset : Int) -> Int {
  read_u32(bytes, offset).reinterpret_as_int()
}

///|
fn read_u64(bytes : FixedArray[Byte], offset : Int) -> UInt64 {
  let mut value = 0UL
  for index in 0..<8 {
    value = value | (bytes[offset + index].to_uint64() << (index * 8))
  }
  value
}

///|
fn frame_checksum(bytes : FixedArray[Byte], length : Int) -> UInt {
  let mut hash : UInt = 2166136261
  for index in 0..<length {
    if index < 16 || index >= 20 {
      hash = (hash ^ bytes[index].to_uint()) * 16777619
    }
  }
  hash
}

///|
fn clear_header(bytes : FixedArray[Byte], length : Int) -> Unit {
  for index in 0..<length {
    bytes[index] = b'\x00'
  }
}

///|
fn copy_bytes(
  source : FixedArray[Byte],
  destination : FixedArray[Byte],
  length : Int,
) -> Unit {
  for index in 0..<length {
    destination[index] = source[index]
  }
}

///|
fn write_bytes(
  destination : FixedArray[Byte],
  offset : Int,
  value : Bytes,
) -> Int {
  for index, byte in value {
    destination[offset + index] = byte
  }
  offset + value.length()
}

///|
fn read_bytes(source : FixedArray[Byte], offset : Int, length : Int) -> Bytes {
  Bytes::makei(length, index => source[offset + index])
}

///|
fn write_digest(
  destination : FixedArray[Byte],
  offset : Int,
  value : String,
  frame : FramedWireFrameKind,
) -> Unit raise FramedWireError {
  let encoded = @utf8.encode(value)
  if encoded.length() != 64 {
    raise InvalidFrame(frame~, rule=Identity)
  }
  ignore(write_bytes(destination, offset, encoded))
}

///|
fn read_digest(
  source : FixedArray[Byte],
  offset : Int,
  frame : FramedWireFrameKind,
) -> String raise FramedWireError {
  @utf8.decode(read_bytes(source, offset, 64)) catch {
    _ => raise InvalidFrame(frame~, rule=Identity)
  }
}
