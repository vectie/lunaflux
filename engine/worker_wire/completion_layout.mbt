///|
let completion_magic : UInt = 0x3258464c

///|
let completion_kind : UInt = 2

///|
let completion_header_bytes : Int = 48

///|
let completion_checksum_offset : Int = 40

///|
let completion_entry_bytes : Int = 40

///|
let max_completion_frame_bytes : Int = 67108864

///|
pub fn completion_frame_byte_capacity(
  limits : @worker_protocol.WorkerProtocolLimits,
) -> Int raise WorkerWireError {
  let count = limits.max_completion_slots()
  if count < 0 ||
    count >
    (max_completion_frame_bytes - completion_header_bytes) /
    completion_entry_bytes {
    raise InvalidLimit(CompletionFrameBytes)
  }
  completion_header_bytes + count * completion_entry_bytes
}

///|
fn exact_completion_frame_bytes(count : Int) -> Int raise WorkerWireError {
  if count < 0 ||
    count >
    (max_completion_frame_bytes - completion_header_bytes) /
    completion_entry_bytes {
    raise InvalidLimit(CompletionFrameBytes)
  }
  completion_header_bytes + count * completion_entry_bytes
}

///|
fn completion_entry_offset(index : Int) -> Int {
  completion_header_bytes + index * completion_entry_bytes
}

///|
fn completion_frame_checksum(bytes : FixedArray[Byte], length : Int) -> UInt {
  let mut hash : UInt = 2166136261
  for index in 0..<length {
    if index < completion_checksum_offset ||
      index >= completion_checksum_offset + 4 {
      hash = (hash ^ bytes[index].to_uint()) * 16777619
    }
  }
  hash
}
