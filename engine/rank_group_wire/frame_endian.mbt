///|
fn wire_write_u32(
  bytes : FixedArray[Byte],
  offset : Int32,
  value : UInt32,
) -> Unit {
  bytes.unsafe_write_uint32_le(offset, value)
}

///|
fn wire_write_i32(
  bytes : FixedArray[Byte],
  offset : Int32,
  value : Int32,
) -> Unit {
  wire_write_u32(bytes, offset, value.reinterpret_as_uint())
}

///|
fn wire_write_u64(
  bytes : FixedArray[Byte],
  offset : Int32,
  value : UInt64,
) -> Unit {
  bytes.unsafe_write_uint64_le(offset, value)
}

///|
fn wire_read_u32(bytes : FixedArray[Byte], offset : Int32) -> UInt32 {
  bytes[offset].to_uint() |
  (bytes[offset + 1].to_uint() << 8) |
  (bytes[offset + 2].to_uint() << 16) |
  (bytes[offset + 3].to_uint() << 24)
}

///|
fn wire_read_i32(bytes : FixedArray[Byte], offset : Int32) -> Int32 {
  wire_read_u32(bytes, offset).reinterpret_as_int()
}

///|
fn wire_read_u64(bytes : FixedArray[Byte], offset : Int32) -> UInt64 {
  let mut result = 0UL
  for index in 0..<8 {
    result = result | (bytes[offset + index].to_uint64() << (index * 8))
  }
  result
}
