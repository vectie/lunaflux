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
fn ChildProcess::clear_pending_write(self : ChildProcess) -> Unit {
  self.write_transfer_active = false
  self.write_transfer_source = self.empty_transfer_buffer
  self.write_transfer_offset = 0
  self.write_transfer_byte_count = 0
  self.write_transfer_deadline_millis = 0L
  self.write_transfer_phase = 0
  self.write_transfer_cursor = 0
}

///|
fn ChildProcess::clear_pending_read(self : ChildProcess) -> Unit {
  self.read_transfer_active = false
  self.read_transfer_destination = self.empty_transfer_buffer
  self.read_transfer_staging = self.empty_transfer_buffer
  self.read_transfer_byte_count = 0
  self.read_transfer_deadline_millis = 0L
  self.read_transfer_phase = 0
  self.read_transfer_cursor = 0
}

///|
fn ChildProcess::clear_pending_transfers(self : ChildProcess) -> Unit {
  self.clear_pending_write()
  self.clear_pending_read()
}

///|
fn ChildProcess::poison_transfers(self : ChildProcess) -> Unit {
  self.failed = true
  self.clear_pending_transfers()
}

///|
fn ChildProcess::require_transfer_unexpired(
  self : ChildProcess,
  deadline_millis : Int64,
  operation : ProcessOperation,
) -> Unit raise ProcessError {
  let now = raw_monotonic_now()
  if now < 0L {
    self.poison_transfers()
    raise OperationFailed(operation)
  }
  if now >= deadline_millis {
    self.poison_transfers()
    raise Timeout(operation)
  }
}

///|
fn ChildProcess::admit_frame_deadline(
  self : ChildProcess,
  deadline_millis : Int64,
  operation : ProcessOperation,
) -> Unit raise ProcessError {
  let now = raw_monotonic_now()
  if now < 0L {
    self.poison_transfers()
    raise OperationFailed(operation)
  }
  if now >= deadline_millis {
    self.poison_transfers()
    raise Timeout(operation)
  }
  if deadline_millis - now > 600000L {
    raise InvalidRange(operation)
  }
}

///|
fn child_transferred_or_pending(
  owner : ChildProcess,
  result : Int,
  operation : ProcessOperation,
) -> Int raise ProcessError {
  if result >= 0 {
    return result
  }
  owner.poison_transfers()
  raise_status(-result, operation)
  abort("unreachable transfer status")
}

///|
fn validate_child_progress_count(
  owner : ChildProcess,
  transferred : Int,
  remaining : Int,
  operation : ProcessOperation,
) -> Unit raise ProcessError {
  if transferred <= 0 || transferred > remaining {
    owner.poison_transfers()
    raise OperationFailed(operation)
  }
}

///|
/// Begins one bounded frame write without performing I/O. One write and one
/// read may be active concurrently on the same child, but a second write is
/// rejected. The deadline covers the complete prefix and payload and is never
/// reset by partial progress. The owner is single-event-loop confined.
pub fn ChildProcess::begin_write_frame(
  self : ChildProcess,
  source : FixedArray[Byte],
  offset~ : Int,
  byte_count~ : Int,
  timeout_millis? : Int = self.limits.io_timeout_millis,
) -> Unit raise ProcessError {
  if timeout_millis <= 0 || timeout_millis > 600000 {
    raise InvalidRange(Write)
  }
  let deadline_millis = capture_deadline(timeout_millis, Write)
  self.begin_write_frame_at_deadline(
    source,
    offset~,
    byte_count~,
    deadline_millis~,
    admit_absolute=false,
  )
}

///|
/// Begins a frame write against one caller-owned absolute monotonic deadline.
/// The same deadline may be reused for the ordered response read.
pub fn ChildProcess::begin_write_frame_until(
  self : ChildProcess,
  source : FixedArray[Byte],
  offset~ : Int,
  byte_count~ : Int,
  deadline_millis~ : Int64,
) -> Unit raise ProcessError {
  self.begin_write_frame_at_deadline(
    source,
    offset~,
    byte_count~,
    deadline_millis~,
    admit_absolute=true,
  )
}

///|
fn ChildProcess::begin_write_frame_at_deadline(
  self : ChildProcess,
  source : FixedArray[Byte],
  offset~ : Int,
  byte_count~ : Int,
  deadline_millis~ : Int64,
  admit_absolute~ : Bool,
) -> Unit raise ProcessError {
  self.require_usable(Write)
  if self.write_transfer_active {
    raise InvalidState(Write)
  }
  if self.write_source_conflicts_active_read(source) {
    raise InvalidRange(Write)
  }
  if byte_count <= 0 ||
    byte_count > self.limits.max_frame_bytes ||
    offset < 0 ||
    offset > source.length() - byte_count {
    raise InvalidRange(Write)
  }
  if admit_absolute {
    self.admit_frame_deadline(deadline_millis, Write)
  }
  write_u32_prefix(self.write_prefix, byte_count)
  self.write_transfer_source = source
  self.write_transfer_offset = offset
  self.write_transfer_byte_count = byte_count
  self.write_transfer_deadline_millis = deadline_millis
  self.write_transfer_phase = 0
  self.write_transfer_cursor = 0
  self.write_transfer_active = true
}

///|
/// Advances one frame write with at most one native `send`. Returns `false`
/// for pending progress and `true` only after the complete payload is sent.
pub fn ChildProcess::progress_write_frame(
  self : ChildProcess,
) -> Bool raise ProcessError {
  self.require_usable(Write)
  if !self.write_transfer_active {
    raise InvalidState(Write)
  }
  self.require_transfer_unexpired(self.write_transfer_deadline_millis, Write)
  let buffer = if self.write_transfer_phase == 0 {
    self.write_prefix
  } else {
    self.write_transfer_source
  }
  let base_offset = if self.write_transfer_phase == 0 {
    0
  } else {
    self.write_transfer_offset
  }
  let phase_count = if self.write_transfer_phase == 0 {
    4
  } else {
    self.write_transfer_byte_count
  }
  let remaining = phase_count - self.write_transfer_cursor
  let result = raw_try_write(
    self.handle,
    buffer,
    base_offset + self.write_transfer_cursor,
    remaining,
  )
  let transferred = child_transferred_or_pending(self, result, Write)
  if transferred == 0 {
    return false
  }
  validate_child_progress_count(self, transferred, remaining, Write)
  self.write_transfer_cursor += transferred
  if self.write_transfer_cursor < phase_count {
    return false
  }
  if self.write_transfer_phase == 0 {
    self.write_transfer_phase = 1
    self.write_transfer_cursor = 0
    return false
  }
  self.clear_pending_write()
  true
}

///|
/// Begins one transactional bounded frame read without performing I/O. The
/// caller supplies separate staging storage at least as large as destination;
/// neither array may be mutated until the transfer completes or fails. One
/// read and one write may be active concurrently, but a second read is
/// rejected. The owner is single-event-loop confined.
pub fn ChildProcess::begin_read_frame(
  self : ChildProcess,
  destination : FixedArray[Byte],
  staging~ : FixedArray[Byte],
  timeout_millis? : Int = self.limits.io_timeout_millis,
) -> Unit raise ProcessError {
  if timeout_millis <= 0 || timeout_millis > 600000 {
    raise InvalidRange(Read)
  }
  let deadline_millis = capture_deadline(timeout_millis, Read)
  self.begin_read_frame_at_deadline(
    destination,
    staging~,
    deadline_millis~,
    admit_absolute=false,
  )
}

///|
/// Begins the ordered response read against the command write's exact
/// absolute monotonic deadline. Expiry poisons the channel before any read.
pub fn ChildProcess::begin_read_frame_until(
  self : ChildProcess,
  destination : FixedArray[Byte],
  staging~ : FixedArray[Byte],
  deadline_millis~ : Int64,
) -> Unit raise ProcessError {
  self.begin_read_frame_at_deadline(
    destination,
    staging~,
    deadline_millis~,
    admit_absolute=true,
  )
}

///|
fn ChildProcess::begin_read_frame_at_deadline(
  self : ChildProcess,
  destination : FixedArray[Byte],
  staging~ : FixedArray[Byte],
  deadline_millis~ : Int64,
  admit_absolute~ : Bool,
) -> Unit raise ProcessError {
  self.require_usable(Read)
  if self.read_transfer_active {
    raise InvalidState(Read)
  }
  if self.read_buffer_conflicts_active_write(destination) ||
    self.read_buffer_conflicts_active_write(staging) {
    raise InvalidRange(Read)
  }
  if destination.length() <= 0 ||
    destination.length() > self.limits.max_frame_bytes ||
    staging.length() < destination.length() ||
    physical_equal(destination, staging) {
    raise InvalidRange(Read)
  }
  if admit_absolute {
    self.admit_frame_deadline(deadline_millis, Read)
  }
  self.read_transfer_destination = destination
  self.read_transfer_staging = staging
  self.read_transfer_byte_count = 0
  self.read_transfer_deadline_millis = deadline_millis
  self.read_transfer_phase = 0
  self.read_transfer_cursor = 0
  self.read_transfer_active = true
}

///|
/// Advances one transactional frame read with at most one native `recv`.
/// `None` is pending; `Some(byte_count)` publishes the complete validated
/// staging range into destination. Destination is unchanged on every failure.
pub fn ChildProcess::progress_read_frame(
  self : ChildProcess,
) -> Int? raise ProcessError {
  self.require_usable(Read)
  if !self.read_transfer_active {
    raise InvalidState(Read)
  }
  self.require_transfer_unexpired(self.read_transfer_deadline_millis, Read)
  let buffer = if self.read_transfer_phase == 0 {
    self.read_prefix
  } else {
    self.read_transfer_staging
  }
  let phase_count = if self.read_transfer_phase == 0 {
    4
  } else {
    self.read_transfer_byte_count
  }
  let remaining = phase_count - self.read_transfer_cursor
  let result = raw_try_read(
    self.handle,
    buffer,
    self.read_transfer_cursor,
    remaining,
  )
  let transferred = child_transferred_or_pending(self, result, Read)
  if transferred == 0 {
    return None
  }
  validate_child_progress_count(self, transferred, remaining, Read)
  self.read_transfer_cursor += transferred
  if self.read_transfer_cursor < phase_count {
    return None
  }
  if self.read_transfer_phase == 0 {
    let byte_count = read_u32_prefix(self.read_prefix)
    if byte_count <= 0 ||
      byte_count > self.limits.max_frame_bytes ||
      byte_count > self.read_transfer_destination.length() ||
      byte_count > self.read_transfer_staging.length() {
      self.poison_transfers()
      raise InvalidRange(Read)
    }
    self.read_transfer_phase = 1
    self.read_transfer_cursor = 0
    self.read_transfer_byte_count = byte_count
    return None
  }
  let byte_count = self.read_transfer_byte_count
  for index in 0..<byte_count {
    self.read_transfer_destination[index] = self.read_transfer_staging[index]
  }
  self.clear_pending_read()
  Some(byte_count)
}
