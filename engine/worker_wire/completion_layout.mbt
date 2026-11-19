///|
let completion_magic : UInt32 = 0x3258464c

///|
let completion_kind : UInt32 = 2

///|
let completion_header_bytes : Int32 = 48

///|
let completion_checksum_offset : Int32 = 40

///|
let completion_entry_bytes : Int32 = 40

///|
let max_completion_frame_bytes : Int32 = 67108864

///|
pub fn completion_frame_byte_capacity(
  limits : @worker_protocol.WorkerProtocolLimits,
) -> Int32 raise WorkerWireError {
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
fn exact_completion_frame_bytes(count : Int32) -> Int32 raise WorkerWireError {
  if count < 0 ||
    count >
    (max_completion_frame_bytes - completion_header_bytes) /
    completion_entry_bytes {
    raise InvalidLimit(CompletionFrameBytes)
  }
  completion_header_bytes + count * completion_entry_bytes
}

///|
fn completion_entry_offset(index : Int32) -> Int32 {
  completion_header_bytes + index * completion_entry_bytes
}
