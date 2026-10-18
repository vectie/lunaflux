///|
/// A request release follows the acknowledged plan predecessor. It is separate
/// from plan completion: cancellation may precede the first GPU submission.
#valtype
pub(all) struct RequestRetirement {
  predecessor : UInt64
  request_id : UInt64
  generation : UInt64
  acknowledged : Bool
} derive(Eq, Debug)

///|
pub fn is_request_retirement(bytes : FixedArray[Byte], length : Int) -> Bool {
  length >= 4 && bytes.length() >= length && read_u32(bytes, 0) == 0x5252464cU
}

///|
pub fn RequestRetirement::encode(
  self : Self,
  destination : FixedArray[Byte],
) -> Unit raise WorkerWireError {
  guard destination.length() >= 40 && self.request_id > 0UL else {
    raise InvalidBootstrapSource(Length)
  }
  write_u32(destination, 0, 0x5252464cU)
  write_u32(destination, 4, 1U)
  write_u64(destination, 8, self.predecessor)
  write_u64(destination, 16, self.request_id)
  write_u64(destination, 24, self.generation)
  write_u32(destination, 32, if self.acknowledged { 1U } else { 0U })
  write_u32(destination, 36, 0U)
}

///|
pub fn decode_request_retirement(
  bytes : FixedArray[Byte],
  length : Int32,
) -> RequestRetirement raise WorkerWireError {
  guard length == 40 &&
    is_request_retirement(bytes, length) &&
    read_u32(bytes, 4) == 1U &&
    read_u32(bytes, 32) <= 1U &&
    read_u32(bytes, 36) == 0U &&
    read_u64(bytes, 16) > 0UL else {
    raise InvalidBootstrapSource(Length)
  }
  {
    predecessor: read_u64(bytes, 8),
    request_id: read_u64(bytes, 16),
    generation: read_u64(bytes, 24),
    acknowledged: read_u32(bytes, 32) == 1U,
  }
}
