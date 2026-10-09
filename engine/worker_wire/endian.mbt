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
  // Carry the scalar in the loop state. A captured mutable accumulator is
  // boxed in native debug builds, allocating once for every wire-field read.
  for index = 0, result = 0UL; index < 8; {
    continue index + 1,
      result | (bytes[offset + index].to_uint64() << (index * 8))
  } nobreak {
    result
  }
}

///|
fn frame_checksum(bytes : FixedArray[Byte], length : Int) -> UInt {
  frame_checksum_at(bytes, 0, length)
}

///|
fn frame_checksum_at(
  bytes : FixedArray[Byte],
  frame_offset : Int,
  length : Int,
) -> UInt {
  for index = 0, hash = 2166136261U; index < length; {
    let next = if index < plan_checksum_offset ||
      index >= plan_checksum_offset + 4 {
      (hash ^ bytes[frame_offset + index].to_uint()) * 16777619U
    } else {
      hash
    }
    continue index + 1, next
  } nobreak {
    hash
  }
}
