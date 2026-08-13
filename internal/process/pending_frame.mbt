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
  self.clear_pending_transfer()
}

///|
fn ChildProcess::clear_pending_transfer(self : ChildProcess) -> Unit {
  self.transfer_kind = 0
  self.transfer_primary = self.empty_transfer_buffer
  self.transfer_secondary = self.empty_transfer_buffer
  self.transfer_offset = 0
  self.transfer_byte_count = 0
  self.transfer_deadline_millis = 0L
  self.transfer_phase = 0
  self.transfer_cursor = 0
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
/// transfer may be active for a child. The deadline covers the complete
/// prefix and payload and is never reset by partial progress. No read or other
/// write transfer may be active for the child.
pub fn ChildProcess::begin_write_frame(
  self : ChildProcess,
  source : FixedArray[Byte],
  offset~ : Int,
  byte_count~ : Int,
  timeout_millis? : Int = self.limits.io_timeout_millis,
) -> Unit raise ProcessError {
  self.require_usable(Write)
  if self.transfer_kind != 0 {
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
  write_u32_prefix(self.write_prefix, byte_count)
  self.transfer_primary = source
  self.transfer_secondary = self.empty_transfer_buffer
  self.transfer_offset = offset
  self.transfer_byte_count = byte_count
  self.transfer_deadline_millis = deadline_millis
  self.transfer_phase = 0
  self.transfer_cursor = 0
  self.transfer_kind = 1
}

///|
/// Advances one frame write with at most one native `send`. Returns `false`
/// for pending progress and `true` only after the complete payload is sent.
pub fn ChildProcess::progress_write_frame(
  self : ChildProcess,
) -> Bool raise ProcessError {
  let owner = self
  owner.require_usable(Write)
  if owner.transfer_kind != 1 {
    raise InvalidState(Write)
  }
  owner.require_unexpired(owner.transfer_deadline_millis, Write)
  let buffer = if owner.transfer_phase == 0 {
    owner.write_prefix
  } else {
    owner.transfer_primary
  }
  let base_offset = if owner.transfer_phase == 0 {
    0
  } else {
    owner.transfer_offset
  }
  let phase_count = if owner.transfer_phase == 0 {
    4
  } else {
    owner.transfer_byte_count
  }
  let remaining = phase_count - owner.transfer_cursor
  let result = raw_try_write(
    owner.handle,
    buffer,
    base_offset + owner.transfer_cursor,
    remaining,
  )
  let transferred = transferred_or_pending(owner, result, Write)
  if transferred == 0 {
    return false
  }
  validate_progress_count(owner, transferred, remaining, Write)
  owner.transfer_cursor += transferred
  if owner.transfer_cursor < phase_count {
    return false
  }
  if owner.transfer_phase == 0 {
    owner.transfer_phase = 1
    owner.transfer_cursor = 0
    return false
  }
  owner.clear_pending_transfer()
  true
}

///|
/// Begins one transactional bounded frame read without performing I/O. The
/// caller supplies separate staging storage at least as large as destination;
/// neither array may be mutated until the transfer completes or fails. No
/// write or other read transfer may be active for the child.
pub fn ChildProcess::begin_read_frame(
  self : ChildProcess,
  destination : FixedArray[Byte],
  staging~ : FixedArray[Byte],
  timeout_millis? : Int = self.limits.io_timeout_millis,
) -> Unit raise ProcessError {
  self.require_usable(Read)
  if self.transfer_kind != 0 {
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
  self.transfer_primary = destination
  self.transfer_secondary = staging
  self.transfer_offset = 0
  self.transfer_byte_count = 0
  self.transfer_deadline_millis = deadline_millis
  self.transfer_phase = 0
  self.transfer_cursor = 0
  self.transfer_kind = 2
}

///|
/// Advances one transactional frame read with at most one native `recv`.
/// `None` is pending; `Some(byte_count)` publishes the complete validated
/// staging range into destination. Destination is unchanged on every failure.
pub fn ChildProcess::progress_read_frame(
  self : ChildProcess,
) -> Int? raise ProcessError {
  let owner = self
  owner.require_usable(Read)
  if owner.transfer_kind != 2 {
    raise InvalidState(Read)
  }
  owner.require_unexpired(owner.transfer_deadline_millis, Read)
  let buffer = if owner.transfer_phase == 0 {
    owner.read_prefix
  } else {
    owner.transfer_secondary
  }
  let phase_count = if owner.transfer_phase == 0 {
    4
  } else {
    owner.transfer_byte_count
  }
  let result = raw_try_read(
    owner.handle,
    buffer,
    owner.transfer_cursor,
    phase_count - owner.transfer_cursor,
  )
  let transferred = transferred_or_pending(owner, result, Read)
  if transferred == 0 {
    return None
  }
  validate_progress_count(
    owner,
    transferred,
    phase_count - owner.transfer_cursor,
    Read,
  )
  owner.transfer_cursor += transferred
  if owner.transfer_cursor < phase_count {
    return None
  }
  if owner.transfer_phase == 0 {
    let byte_count = read_u32_prefix(owner.read_prefix)
    if byte_count <= 0 ||
      byte_count > owner.limits.max_frame_bytes ||
      byte_count > owner.transfer_primary.length() ||
      byte_count > owner.transfer_secondary.length() {
      owner.poison_transfer()
      raise InvalidRange(Read)
    }
    owner.transfer_phase = 1
    owner.transfer_cursor = 0
    owner.transfer_byte_count = byte_count
    return None
  }
  let byte_count = owner.transfer_byte_count
  for index in 0..<byte_count {
    owner.transfer_primary[index] = owner.transfer_secondary[index]
  }
  owner.clear_pending_transfer()
  Some(byte_count)
}
