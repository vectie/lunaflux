///|
fn configure_write_u32(
  bytes : FixedArray[Byte],
  offset : Int,
  value : UInt,
) -> Unit {
  bytes[offset] = value.to_byte()
  bytes[offset + 1] = (value >> 8).to_byte()
  bytes[offset + 2] = (value >> 16).to_byte()
  bytes[offset + 3] = (value >> 24).to_byte()
}

///|
fn configure_read_u32(bytes : FixedArray[Byte], offset : Int) -> UInt {
  bytes[offset].to_uint() |
  (bytes[offset + 1].to_uint() << 8) |
  (bytes[offset + 2].to_uint() << 16) |
  (bytes[offset + 3].to_uint() << 24)
}

///|
fn configure_write_i32(
  bytes : FixedArray[Byte],
  offset : Int,
  value : Int,
) -> Unit {
  configure_write_u32(bytes, offset, value.reinterpret_as_uint())
}

///|
fn configure_read_i32(bytes : FixedArray[Byte], offset : Int) -> Int {
  configure_read_u32(bytes, offset).reinterpret_as_int()
}

///|
fn configure_write_u64(
  bytes : FixedArray[Byte],
  offset : Int,
  value : UInt64,
) -> Unit {
  for index in 0..<8 {
    bytes[offset + index] = (value >> (index * 8)).to_byte()
  }
}

///|
fn configure_read_u64(bytes : FixedArray[Byte], offset : Int) -> UInt64 {
  let mut value = 0UL
  for index in 0..<8 {
    value = value | (bytes[offset + index].to_uint64() << (index * 8))
  }
  value
}

///|
fn configure_checksum(
  bytes : FixedArray[Byte],
  start : Int,
  length : Int,
  checksum_offset : Int,
) -> UInt {
  let mut hash : UInt = 2166136261
  for index in 0..<length {
    let absolute = start + index
    if absolute < checksum_offset || absolute >= checksum_offset + 4 {
      hash = (hash ^ bytes[absolute].to_uint()) * 16777619
    }
  }
  hash
}

///|
fn configure_hex_nibble(
  unit : UInt16,
) -> Byte raise TensorParallelRankConfigureError {
  if unit >= 48 && unit <= 57 {
    (unit.to_int() - '0'.to_int()).to_byte()
  } else if unit >= 97 && unit <= 102 {
    (unit.to_int() - 'a'.to_int() + 10).to_byte()
  } else {
    raise Invalid(Digest)
  }
}

///|
fn configure_write_raw_digest(
  destination : FixedArray[Byte],
  offset : Int,
  digest : StringView,
) -> Unit raise TensorParallelRankConfigureError {
  if digest.length() != 64 {
    raise Invalid(Digest)
  }
  for index in 0..<32 {
    destination[offset + index] = (configure_hex_nibble(digest[index * 2]) << 4) |
      configure_hex_nibble(digest[index * 2 + 1])
  }
}

///|
fn configure_read_raw_digest(
  source : FixedArray[Byte],
  offset : Int,
) -> TensorParallelRankConfigureDigest {
  TensorParallelRankConfigureDigest(
    @crypto.bytes_to_hex_string(
      Bytes::makei(32, index => source[offset + index]),
    ),
  )
}
