///|
pub fn frame_bytes() -> Int {
  64
}

///|
pub fn is_frame(bytes : FixedArray[Byte], length : Int) -> Bool {
  length == 64 && bytes.length() >= 64 && read32(bytes, 0) == 0x5254464c
}

///|
fn read32(bytes : FixedArray[Byte], offset : Int) -> Int {
  (bytes[offset].to_uint() |
  (bytes[offset + 1].to_uint() << 8) |
  (bytes[offset + 2].to_uint() << 16) |
  (bytes[offset + 3].to_uint() << 24)).reinterpret_as_int()
}

///|
fn read64(bytes : FixedArray[Byte], offset : Int) -> UInt64 {
  for i in 0..<8; result = 0UL {
    continue result | (bytes[offset + i].to_uint64() << (i * 8))
  } nobreak {
    result
  }
}

///|
pub fn encode(
  reply : Reply,
  bytes : FixedArray[Byte],
) -> Unit raise StreamingError {
  if bytes.length() < 64 {
    raise InvalidFrame
  }
  for i in 0..<64 {
    bytes[i] = b'\x00'
  }
  bytes.unsafe_write_uint32_le(0, 0x5254464c)
  bytes.unsafe_write_uint32_le(4, 1)
  bytes.unsafe_write_uint64_le(8, reply.command.worker)
  bytes.unsafe_write_uint64_le(16, reply.command.sequence)
  let opcode = match reply.command.opcode {
    Spill => 1
    Restore => 2
    Poll => 3
    Cancel => 4
    Evict => 5
  }
  let status = match reply.status {
    Pending => 0
    HostReady => 1
    DeviceReady => 2
    Cancelled => 3
    Evicted => 4
    Rejected => 5
  }
  bytes.unsafe_write_uint32_le(24, opcode.reinterpret_as_uint())
  bytes.unsafe_write_uint32_le(28, reply.command.page.reinterpret_as_uint())
  bytes.unsafe_write_uint32_le(
    32,
    reply.command.page_generation.reinterpret_as_uint(),
  )
  bytes.unsafe_write_uint32_le(
    36,
    reply.command.host.slot.reinterpret_as_uint(),
  )
  bytes.unsafe_write_uint32_le(
    40,
    reply.command.host.generation.reinterpret_as_uint(),
  )
  bytes.unsafe_write_uint32_le(44, status.reinterpret_as_uint())
  bytes.unsafe_write_uint32_le(48, reply.host.slot.reinterpret_as_uint())
  bytes.unsafe_write_uint32_le(52, reply.host.generation.reinterpret_as_uint())
}

///|
pub fn decode(
  bytes : FixedArray[Byte],
  length : Int,
) -> Reply raise StreamingError {
  if !is_frame(bytes, length) ||
    read32(bytes, 4) != 1 ||
    read64(bytes, 56) != 0UL {
    raise InvalidFrame
  }
  let opcode = match read32(bytes, 24) {
    1 => Spill
    2 => Restore
    3 => Poll
    4 => Cancel
    5 => Evict
    _ => raise InvalidFrame
  }
  let status = match read32(bytes, 44) {
    0 => Pending
    1 => HostReady
    2 => DeviceReady
    3 => Cancelled
    4 => Evicted
    5 => Rejected
    _ => raise InvalidFrame
  }
  {
    command: {
      worker: read64(bytes, 8),
      sequence: read64(bytes, 16),
      opcode,
      page: read32(bytes, 28),
      page_generation: read32(bytes, 32),
      host: { slot: read32(bytes, 36), generation: read32(bytes, 40), },
    },
    status,
    host: { slot: read32(bytes, 48), generation: read32(bytes, 52), },
  }
}
