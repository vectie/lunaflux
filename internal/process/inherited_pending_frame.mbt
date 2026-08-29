///|
fn InheritedChannel::clear_pending_write(self : InheritedChannel) -> Unit {
  self.write_transfer_active = false
  self.write_transfer_source = self.empty_transfer_buffer
  self.write_transfer_offset = 0
  self.write_transfer_byte_count = 0
  self.write_transfer_deadline_millis = 0L
  self.write_transfer_phase = 0
  self.write_transfer_cursor = 0
}

///|
fn InheritedChannel::clear_pending_read(self : InheritedChannel) -> Unit {
  self.read_transfer_active = false
  self.read_transfer_destination = self.empty_transfer_buffer
  self.read_transfer_staging = self.empty_transfer_buffer
  self.read_transfer_byte_count = 0
  self.read_transfer_deadline_millis = 0L
  self.read_transfer_phase = 0
  self.read_transfer_cursor = 0
  self.read_transfer_allows_eof = false
}

///|
fn InheritedChannel::clear_pending_transfers(self : InheritedChannel) -> Unit {
  self.clear_pending_write()
  self.clear_pending_read()
}

///|
fn InheritedChannel::poison_transfers(self : InheritedChannel) -> Unit {
  self.failed = true
  self.clear_pending_transfers()
}

///|
fn InheritedChannel::require_transfer_unexpired(
  self : InheritedChannel,
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
fn inherited_transferred_or_pending(
  owner : InheritedChannel,
  result : Int,
  operation : ProcessOperation,
) -> Int raise ProcessError {
  if result >= 0 {
    return result
  }
  owner.poison_transfers()
  raise_status(-result, operation)
  abort("unreachable inherited transfer status")
}

///|
fn validate_inherited_progress_count(
  owner : InheritedChannel,
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
/// Begins one cooperative worker-side frame write without performing I/O.
/// One read may coexist with the write, while another write is rejected. The
/// channel is single-event-loop confined and the deadline is not refreshed.
pub fn InheritedChannel::begin_write_frame(
  self : InheritedChannel,
  source : FixedArray[Byte],
  offset~ : Int,
  byte_count~ : Int,
  timeout_millis? : Int = self.limits.io_timeout_millis,
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
    timeout_millis <= 0 ||
    timeout_millis > 600000 ||
    offset < 0 ||
    offset > source.length() - byte_count {
    raise InvalidRange(Write)
  }
  let deadline_millis = capture_deadline(timeout_millis, Write)
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
/// Advances one cooperative worker-side frame write with at most one native
/// `write`. Returns true only after the complete prefix and payload are sent.
pub fn InheritedChannel::progress_write_frame(
  self : InheritedChannel,
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
  let result = raw_inherited_try_write(
    buffer,
    base_offset + self.write_transfer_cursor,
    remaining,
  )
  let transferred = inherited_transferred_or_pending(self, result, Write)
  if transferred == 0 {
    return false
  }
  validate_inherited_progress_count(self, transferred, remaining, Write)
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
fn InheritedChannel::begin_read_frame_internal(
  self : InheritedChannel,
  destination : FixedArray[Byte],
  staging : FixedArray[Byte],
  timeout_millis : Int,
  allows_eof : Bool,
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
    physical_equal(destination, staging) ||
    timeout_millis <= 0 ||
    timeout_millis > 600000 {
    raise InvalidRange(Read)
  }
  let deadline_millis = capture_deadline(timeout_millis, Read)
  self.read_transfer_destination = destination
  self.read_transfer_staging = staging
  self.read_transfer_byte_count = 0
  self.read_transfer_deadline_millis = deadline_millis
  self.read_transfer_phase = 0
  self.read_transfer_cursor = 0
  self.read_transfer_allows_eof = allows_eof
  self.read_transfer_active = true
}

///|
/// Begins one cooperative transactional worker-side frame read. One write may
/// coexist with the read, while another read is rejected. EOF is an error.
pub fn InheritedChannel::begin_read_frame(
  self : InheritedChannel,
  destination : FixedArray[Byte],
  staging~ : FixedArray[Byte],
  timeout_millis? : Int = self.limits.io_timeout_millis,
) -> Unit raise ProcessError {
  self.begin_read_frame_internal(destination, staging, timeout_millis, false)
}

///|
/// Begins one cooperative transactional worker-side frame read that may
/// publish clean EOF only before the first byte of a new prefix.
pub fn InheritedChannel::begin_read_frame_or_eof(
  self : InheritedChannel,
  destination : FixedArray[Byte],
  staging~ : FixedArray[Byte],
  timeout_millis? : Int = self.limits.io_timeout_millis,
) -> Unit raise ProcessError {
  self.begin_read_frame_internal(destination, staging, timeout_millis, true)
}

///|
fn InheritedChannel::progress_read_frame_internal(
  self : InheritedChannel,
  expects_eof_mode : Bool,
) -> Int? raise ProcessError {
  self.require_usable(Read)
  if !self.read_transfer_active ||
    self.read_transfer_allows_eof != expects_eof_mode {
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
  let result = raw_inherited_try_read(
    buffer,
    self.read_transfer_cursor,
    remaining,
  )
  if result == -3 &&
    self.read_transfer_allows_eof &&
    self.read_transfer_phase == 0 &&
    self.read_transfer_cursor == 0 {
    self.clear_pending_read()
    return Some(0)
  }
  let transferred = inherited_transferred_or_pending(self, result, Read)
  if transferred == 0 {
    return None
  }
  validate_inherited_progress_count(self, transferred, remaining, Read)
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

///|
/// Advances a cooperative worker-side frame read with at most one native
/// `read`. None is pending and a positive value is a complete frame length.
pub fn InheritedChannel::progress_read_frame(
  self : InheritedChannel,
) -> Int? raise ProcessError {
  self.progress_read_frame_internal(false)
}

///|
/// Advances a cooperative worker-side frame-or-EOF read with at most one
/// native `read`. `Some(0)` is clean EOF before a prefix, a positive value is
/// a complete frame length, and `None` is pending.
pub fn InheritedChannel::progress_read_frame_or_eof(
  self : InheritedChannel,
) -> Int? raise ProcessError {
  self.progress_read_frame_internal(true)
}
