///|
pub(all) suberror StageControlError {
  InvalidConfiguration
  InvalidState
  InvalidMessage
  Failed
}

///|
// Control is separate from activation bytes. Existing canonical plan frames
// carry request/model identity; this small envelope carries pipeline step order.
let control_header_bytes : Int = 24

///|
let prepare_kind : Int = 1

///|
let inputs_kind : Int = 2

///|
let run_kind : Int = 3

///|
let result_kind : Int = 4

///|
let commit_kind : Int = 5

///|
let committed_kind : Int = 6

///|
let release_kind : Int = 7

///|
let released_kind : Int = 8

///|
/// Fixed byte/integer storage ceiling for a server or client. Object/socket
/// runtime overhead and rank/device/activation storage are reserved separately.
pub fn required_buffer_bytes(
  limits : @protocol.WorkerProtocolLimits,
) -> Int64 raise {
  let plan = @wire.plan_frame_byte_capacity(limits)
  let completion = @wire.completion_frame_byte_capacity(limits)
  guard plan <= 67108864 - control_header_bytes else {
    raise StageControlError::InvalidConfiguration
  }
  5L * plan.to_int64() +
  completion.to_int64() +
  80L +
  12L * limits.max_plan_rows().to_int64()
}

///|
fn write_u64(bytes : FixedArray[Byte], offset : Int, value : UInt64) -> Unit {
  for i in 0..<8 {
    bytes[offset + i] = ((value >> (8 * i)) & 255UL).to_byte()
  }
}

///|
fn read_u64(bytes : FixedArray[Byte], offset : Int) -> UInt64 {
  for i in 0..<8; result = 0UL {
    continue result | (bytes[offset + i].to_uint64() << (8 * i))
  } nobreak {
    result
  }
}

///|
fn write_i32(bytes : FixedArray[Byte], offset : Int, value : Int) -> Unit {
  let bits = value.reinterpret_as_uint()
  for i in 0..<4 {
    bytes[offset + i] = ((bits >> (8 * i)) & 255U).to_byte()
  }
}

///|
fn read_i32(bytes : FixedArray[Byte], offset : Int) -> Int {
  for i in 0..<4; result = 0U {
    continue result | (bytes[offset + i].to_uint() << (8 * i))
  } nobreak {
    result.reinterpret_as_int()
  }
}

///|
fn header(
  bytes : FixedArray[Byte],
  kind : Int,
  epoch : UInt64,
  flags : Int,
  token : Int,
) -> Unit {
  write_i32(bytes, 0, kind)
  write_i32(bytes, 4, flags)
  write_u64(bytes, 8, epoch)
  write_i32(bytes, 16, token)
  // Reserved word prevents accidental interpretation of a different schema.
  write_i32(bytes, 20, 0)
}

///|
fn validate_header(bytes : FixedArray[Byte], length : Int) -> Unit raise {
  guard length >= control_header_bytes && read_i32(bytes, 20) == 0 else {
    raise StageControlError::InvalidMessage
  }
}
