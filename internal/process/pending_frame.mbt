///|
let max_transfer_epoch : UInt64 = 0xffffffffffffffffUL

///|
fn ChildProcess::claim_transfer_epoch(
  self : ChildProcess,
  operation : ProcessOperation,
) -> UInt64 raise ProcessError {
  if self.next_transfer_epoch == max_transfer_epoch {
    raise OperationFailed(operation)
  }
  let epoch = self.next_transfer_epoch
  self.next_transfer_epoch = epoch + 1UL
  epoch
}

///|
fn capture_deadline(
  timeout_millis : Int,
  operation : ProcessOperation,
) -> Int64 raise ProcessError {
  let now = raw_monotonic_now()
  if now < 0L || now > 0x7fffffffffffffffL - timeout_millis.to_int64() {
    raise OperationFailed(operation)
  }
  now + timeout_millis.to_int64()
}

///|
fn ChildProcess::poison_transfer(self : ChildProcess) -> Unit {
  self.failed = true
  self.active_write_epoch = 0UL
  self.active_read_epoch = 0UL
}

///|
fn ChildProcess::require_unexpired(
  self : ChildProcess,
  deadline_millis : Int64,
  operation : ProcessOperation,
) -> Unit raise ProcessError {
  let now = raw_monotonic_now()
  if now < 0L {
    self.poison_transfer()
    raise OperationFailed(operation)
  }
  if now >= deadline_millis {
    self.poison_transfer()
    raise Timeout(operation)
  }
}

///|
fn transferred_or_pending(
  owner : ChildProcess,
  result : Int,
  operation : ProcessOperation,
) -> Int raise ProcessError {
  if result >= 0 {
    return result
  }
  owner.poison_transfer()
  raise_status(-result, operation)
  abort("unreachable transfer status")
}

///|
fn validate_progress_count(
  owner : ChildProcess,
  transferred : Int,
  remaining : Int,
  operation : ProcessOperation,
) -> Unit raise ProcessError {
  if transferred <= 0 || transferred > remaining {
    owner.poison_transfer()
    raise OperationFailed(operation)
  }
}

///|
/// Begins one bounded frame write without performing I/O. At most one write
/// capability may be active for a child. The deadline covers the complete
/// prefix and payload and is never reset by partial progress. No read or other
/// write capability may be active for the child.
pub fn ChildProcess::begin_write_frame(
  self : ChildProcess,
  source : FixedArray[Byte],
  offset~ : Int,
  byte_count~ : Int,
  timeout_millis? : Int = self.limits.io_timeout_millis,
) -> PendingFrameWrite raise ProcessError {
  self.require_usable(Write)
  if self.active_write_epoch != 0UL || self.active_read_epoch != 0UL {
    raise InvalidState(Write)
  }
  if byte_count <= 0 ||
    byte_count > self.limits.max_frame_bytes ||
    timeout_millis <= 0 ||
    timeout_millis > 600000 ||
    offset < 0 ||
    offset > source.length() - byte_count {
    raise InvalidRange(Write)
  }
  let deadline_millis = capture_deadline(timeout_millis, Write)
  let epoch = self.claim_transfer_epoch(Write)
  write_u32_prefix(self.write_prefix, byte_count)
  self.active_write_epoch = epoch
  {
    owner: self,
    epoch,
    source,
    source_offset: offset,
    byte_count,
    deadline_millis,
    phase: 0,
    cursor: 0,
  }
}

///|
/// Advances one frame write with at most one native `send`. Returns `false`
/// for pending progress and `true` only after the complete payload is sent.
pub fn PendingFrameWrite::progress(
  self : PendingFrameWrite,
) -> Bool raise ProcessError {
  let owner = self.owner
  owner.require_usable(Write)
  if self.epoch == 0UL || owner.active_write_epoch != self.epoch {
    raise InvalidState(Write)
  }
  owner.require_unexpired(self.deadline_millis, Write)
  let buffer = if self.phase == 0 { owner.write_prefix } else { self.source }
  let base_offset = if self.phase == 0 { 0 } else { self.source_offset }
  let phase_count = if self.phase == 0 { 4 } else { self.byte_count }
  let remaining = phase_count - self.cursor
  let result = raw_try_write(
    owner.handle,
    buffer,
    base_offset + self.cursor,
    remaining,
  )
  let transferred = transferred_or_pending(owner, result, Write)
  if transferred == 0 {
    return false
  }
  validate_progress_count(owner, transferred, remaining, Write)
  self.cursor += transferred
  if self.cursor < phase_count {
    return false
  }
  if self.phase == 0 {
    self.phase = 1
    self.cursor = 0
    return false
  }
  owner.active_write_epoch = 0UL
  true
}

///|
/// Begins one transactional bounded frame read without performing I/O. The
/// caller supplies separate staging storage at least as large as destination;
/// neither array may be mutated until the capability completes or fails. No
/// write or other read capability may be active for the child.
pub fn ChildProcess::begin_read_frame(
  self : ChildProcess,
  destination : FixedArray[Byte],
  staging~ : FixedArray[Byte],
  timeout_millis? : Int = self.limits.io_timeout_millis,
) -> PendingFrameRead raise ProcessError {
  self.require_usable(Read)
  if self.active_write_epoch != 0UL || self.active_read_epoch != 0UL {
    raise InvalidState(Read)
  }
  if destination.length() <= 0 ||
    destination.length() > self.limits.max_frame_bytes ||
    staging.length() < destination.length() ||
    physical_equal(destination, staging) ||
    timeout_millis <= 0 ||
    timeout_millis > 600000 {
    raise InvalidRange(Read)
  }
  let deadline_millis = capture_deadline(timeout_millis, Read)
  let epoch = self.claim_transfer_epoch(Read)
  self.active_read_epoch = epoch
  {
    owner: self,
    epoch,
    destination,
    staging,
    deadline_millis,
    phase: 0,
    cursor: 0,
    byte_count: 0,
  }
}

///|
/// Advances one transactional frame read with at most one native `recv`.
/// `None` is pending; `Some(byte_count)` publishes the complete validated
/// staging range into destination. Destination is unchanged on every failure.
pub fn PendingFrameRead::progress(
  self : PendingFrameRead,
) -> Int? raise ProcessError {
  let owner = self.owner
  owner.require_usable(Read)
  if self.epoch == 0UL || owner.active_read_epoch != self.epoch {
    raise InvalidState(Read)
  }
  owner.require_unexpired(self.deadline_millis, Read)
  let buffer = if self.phase == 0 { owner.read_prefix } else { self.staging }
  let phase_count = if self.phase == 0 { 4 } else { self.byte_count }
  let result = raw_try_read(
    owner.handle,
    buffer,
    self.cursor,
    phase_count - self.cursor,
  )
  let transferred = transferred_or_pending(owner, result, Read)
  if transferred == 0 {
    return None
  }
  validate_progress_count(owner, transferred, phase_count - self.cursor, Read)
  self.cursor += transferred
  if self.cursor < phase_count {
    return None
  }
  if self.phase == 0 {
    let byte_count = read_u32_prefix(owner.read_prefix)
    if byte_count <= 0 ||
      byte_count > owner.limits.max_frame_bytes ||
      byte_count > self.destination.length() ||
      byte_count > self.staging.length() {
      owner.poison_transfer()
      raise InvalidRange(Read)
    }
    self.phase = 1
    self.cursor = 0
    self.byte_count = byte_count
    return None
  }
  for index in 0..<self.byte_count {
    self.destination[index] = self.staging[index]
  }
  owner.active_read_epoch = 0UL
  Some(self.byte_count)
}
