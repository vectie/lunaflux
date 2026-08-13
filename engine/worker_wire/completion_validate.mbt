///|
fn validate_completion_frame_source(
  bytes : FixedArray[Byte],
  supplied_length : Int,
  worker_limits : @worker_protocol.WorkerProtocolLimits,
  inference_limits : @inference.InferenceLimits,
  destination_capacity : Int,
  validation_requests : FixedArray[UInt64],
  validation_slots : FixedArray[Int],
) -> Int raise WorkerWireError {
  if supplied_length < completion_header_bytes ||
    supplied_length > bytes.length() {
    raise SourceTooSmall
  }
  if supplied_length > destination_capacity {
    raise InvalidLimit(CompletionFrameBytes)
  }
  if read_u32(bytes, 0) != completion_magic {
    raise InvalidFrame(Magic)
  }
  if read_u32(bytes, 4) != wire_version {
    raise InvalidFrame(Version)
  }
  if read_u32(bytes, 8) != completion_kind {
    raise InvalidFrame(Kind)
  }
  if read_i32(bytes, 12) != supplied_length {
    raise InvalidFrame(Length)
  }
  if read_u32(bytes, 36) != 0 || read_u32(bytes, 44) != 0 {
    raise InvalidFrame(Reserved)
  }
  if completion_frame_checksum(bytes, supplied_length) !=
    read_u32(bytes, completion_checksum_offset) {
    raise InvalidFrame(Checksum)
  }
  if read_u64(bytes, 16) == 0UL || read_u64(bytes, 24) == 0UL {
    raise InvalidFrame(RequestIdentity)
  }
  let count = read_i32(bytes, 32)
  if count <= 0 || count > worker_limits.max_completion_slots() {
    raise InvalidFrame(Counts)
  }
  if exact_completion_frame_bytes(count) != supplied_length {
    raise InvalidFrame(Length)
  }
  let max_token = if worker_limits.max_token_id() <
    inference_limits.max_token_id() {
    worker_limits.max_token_id()
  } else {
    inference_limits.max_token_id()
  }
  for index in 0..<count {
    let offset = completion_entry_offset(index)
    let slot = read_i32(bytes, offset)
    let kind = read_u32(bytes, offset + 4)
    let request = read_u64(bytes, offset + 8)
    let processed = read_i32(bytes, offset + 24)
    let token = read_i32(bytes, offset + 28)
    let failure = read_u32(bytes, offset + 32)
    if slot < 0 || slot >= worker_limits.max_completion_slots() {
      raise InvalidFrame(CompletionSlot)
    }
    if request == 0UL {
      raise InvalidFrame(RequestIdentity)
    }
    if read_u32(bytes, offset + 36) != 0 {
      raise InvalidFrame(Reserved)
    }
    match kind {
      0 =>
        if processed <= 0 ||
          processed > worker_limits.max_plan_tokens() ||
          token != 0 ||
          failure != 0 {
          raise InvalidFrame(CompletionEntry)
        }
      1 =>
        if processed <= 0 ||
          processed > worker_limits.max_plan_tokens() ||
          token < 0 ||
          token > max_token ||
          failure != 0 {
          raise InvalidFrame(CompletionEntry)
        }
      2 =>
        if processed != 0 || token < 0 || token > max_token || failure != 0 {
          raise InvalidFrame(CompletionEntry)
        }
      3 =>
        if processed != 0 || token != 0 || failure > 2 {
          raise InvalidFrame(CompletionEntry)
        }
      _ => raise InvalidFrame(CompletionEntry)
    }
    validation_requests[index] = request
    validation_slots[index] = slot
  }
  validate_unique_scalar_identities(
    validation_requests, validation_slots, count,
  )
  count
}

///|
/// Validates an untrusted completion frame before replacing the current view.
pub fn CompletionFrameBuffer::load(
  self : CompletionFrameBuffer,
  source : FixedArray[Byte],
  length : Int,
) -> ValidatedCompletionFrame raise WorkerWireError {
  if self.epoch == 0xffffffffffffffffUL {
    raise FrameEpochOverflow
  }
  let count = validate_completion_frame_source(
    source,
    length,
    self.worker_limits,
    self.inference_limits,
    self.storage.length(),
    self.validation_requests,
    self.validation_slots,
  )
  self.clear_slot_index()
  for index in 0..<length {
    self.storage[index] = source[index]
  }
  self.epoch += 1UL
  self.length = length
  self.rebuild_slot_index(count)
  ValidatedCompletionFrame(self, self.epoch)
}

///|
/// Copies the authenticated frame to caller-owned transport storage.
pub fn ValidatedCompletionFrame::copy_to(
  self : ValidatedCompletionFrame,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Unit raise WorkerWireError {
  let owner = self.owner()
  if destination_offset < 0 ||
    destination_offset > destination.length() ||
    owner.length > destination.length() - destination_offset {
    raise DestinationTooSmall
  }
  for index in 0..<owner.length {
    destination[destination_offset + index] = owner.storage[index]
  }
}
