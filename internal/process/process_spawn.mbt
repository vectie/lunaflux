///|
/// Allocates every managed field of one child owner before native spawn or an
/// external capability borrow.
pub fn prepare_child(
  executable : Bytes,
  limits : ProcessLimits,
) -> PreparedChildProcess raise ProcessError {
  if executable.length() == 0 || executable.length() > 1048575 {
    raise InvalidRange(Close)
  }
  for index = 0; index < executable.length(); index = index + 1 {
    if executable[index] == b'\x00' {
      raise InvalidRange(Close)
    }
  }
  let path = Bytes::makei(executable.length() + 1, index => {
    if index == executable.length() {
      b'\x00'
    } else {
      executable[index]
    }
  })
  let empty_transfer_buffer = FixedArray::make(0, b'\x00')
  {
    child: {
      handle: raw_prepare_child(),
      limits,
      write_prefix: FixedArray::make(4, b'\x00'),
      read_prefix: FixedArray::make(4, b'\x00'),
      transfer_kind: 0,
      empty_transfer_buffer,
      transfer_primary: empty_transfer_buffer,
      transfer_secondary: empty_transfer_buffer,
      transfer_offset: 0,
      transfer_byte_count: 0,
      transfer_deadline_millis: 0L,
      transfer_phase: 0,
      transfer_cursor: 0,
      failed: false,
      closed: true,
    },
    path,
    consumed: false,
    cleanup_status: 0,
  }
}

///|
fn PreparedChildProcess::require_fresh(
  self : PreparedChildProcess,
) -> Unit raise ProcessError {
  if self.consumed {
    raise InvalidState(Close)
  }
}

///|
pub fn PreparedChildProcess::spawn(
  self : PreparedChildProcess,
) -> ChildProcess raise ProcessError {
  self.require_fresh()
  self.consumed = true
  if raw_spawn_prepared(self.child.handle, self.path) != 0 {
    ignore(raw_close(self.child.handle))
    raise SpawnFailed
  }
  self.child.closed = false
  self.child
}

///|
pub fn spawn(
  executable : Bytes,
  limits : ProcessLimits,
) -> ChildProcess raise ProcessError {
  prepare_child(executable, limits).spawn()
}

///|
pub fn spawn_with_approved_roots(
  executable : Bytes,
  roots : @inheritance.WorkerApprovedRoots,
  limits : ProcessLimits,
) -> ChildProcess raise ProcessError {
  prepare_child(executable, limits).spawn_with_approved_roots(roots)
}

///|
pub fn PreparedChildProcess::spawn_with_approved_roots(
  self : PreparedChildProcess,
  roots : @inheritance.WorkerApprovedRoots,
) -> ChildProcess raise ProcessError {
  self.require_fresh()
  self.consumed = true
  if raw_spawn_prepared_with_approved_roots(self.child.handle, self.path, roots) !=
    0 {
    ignore(raw_close(self.child.handle))
    raise SpawnFailed
  }
  self.child.closed = false
  self.child
}

///|
pub fn PreparedChildProcess::spawn_with_approved_roots_status(
  self : PreparedChildProcess,
  roots : @inheritance.WorkerApprovedRoots,
) -> Int {
  if self.consumed {
    return 1
  }
  self.consumed = true
  let status = raw_spawn_prepared_with_approved_roots(
    self.child.handle,
    self.path,
    roots,
  )
  if status != 0 {
    let cleanup = raw_close(self.child.handle)
    self.cleanup_status = cleanup
    self.child.closed = raw_is_closed(self.child.handle)
    self.child.failed = true
    if !self.child.closed || cleanup != 0 {
      return 4
    }
    return status
  }
  self.child.closed = false
  0
}

///|
pub fn ChildProcess::startup_write_frame_status(
  self : ChildProcess,
  source : FixedArray[Byte],
  byte_count : Int,
  timeout_millis : Int,
) -> Int {
  if self.closed ||
    self.failed ||
    self.transfer_kind != 0 ||
    byte_count <= 0 ||
    byte_count > self.limits.max_frame_bytes ||
    byte_count > source.length() ||
    timeout_millis <= 0 ||
    timeout_millis > 600000 {
    return 1
  }
  write_u32_prefix(self.write_prefix, byte_count)
  let prefix = raw_write_exact(
    self.handle,
    self.write_prefix,
    0,
    4,
    timeout_millis,
  )
  if prefix != 0 {
    self.failed = true
    return prefix
  }
  let payload = raw_write_exact(
    self.handle,
    source,
    0,
    byte_count,
    timeout_millis,
  )
  if payload != 0 {
    self.failed = true
  }
  payload
}

///|
pub fn ChildProcess::startup_read_frame_status(
  self : ChildProcess,
  destination : FixedArray[Byte],
  timeout_millis : Int,
) -> Int {
  if self.closed ||
    self.failed ||
    self.transfer_kind != 0 ||
    timeout_millis <= 0 ||
    timeout_millis > 600000 {
    return -2
  }
  let prefix = raw_read_exact(
    self.handle,
    self.read_prefix,
    0,
    4,
    timeout_millis,
  )
  if prefix != 0 {
    self.failed = true
    return -(prefix + 1)
  }
  let byte_count = read_u32_prefix(self.read_prefix)
  if byte_count <= 0 ||
    byte_count > self.limits.max_frame_bytes ||
    byte_count > destination.length() {
    self.failed = true
    return -2
  }
  let payload = raw_read_exact(
    self.handle,
    destination,
    0,
    byte_count,
    timeout_millis,
  )
  if payload != 0 {
    self.failed = true
    return -(payload + 1)
  }
  byte_count
}

///|
pub fn ChildProcess::startup_close_status(self : ChildProcess) -> Int {
  if self.closed {
    return 0
  }
  self.clear_pending_transfer()
  let status = raw_close(self.handle)
  self.closed = raw_is_closed(self.handle)
  if status != 0 {
    self.failed = true
  }
  status
}
