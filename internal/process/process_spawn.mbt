///|
/// Allocates every managed field of one child owner before native spawn or an
/// external capability borrow.
fn prepare_approved_child(
  approved_executable : @executable_capability.ApprovedExecutableHandle,
  limits : ProcessLimits,
) -> PreparedChildProcess {
  let empty_transfer_buffer = FixedArray::make(0, b'\x00')
  {
    child: {
      handle: raw_prepare_child(),
      limits,
      write_prefix: FixedArray::make(4, b'\x00'),
      read_prefix: FixedArray::make(4, b'\x00'),
      empty_transfer_buffer,
      write_transfer_active: false,
      write_transfer_source: empty_transfer_buffer,
      write_transfer_offset: 0,
      write_transfer_byte_count: 0,
      write_transfer_deadline_millis: 0L,
      write_transfer_phase: 0,
      write_transfer_cursor: 0,
      read_transfer_active: false,
      read_transfer_destination: empty_transfer_buffer,
      read_transfer_staging: empty_transfer_buffer,
      read_transfer_byte_count: 0,
      read_transfer_deadline_millis: 0L,
      read_transfer_phase: 0,
      read_transfer_cursor: 0,
      maintenance_phase: 0,
      maintenance_phase_started_millis: 0L,
      maintenance_deadline_millis: 0L,
      maintenance_resume_phase: 0,
      maintenance_exit_kind: 0,
      maintenance_exit_code: 0,
      maintenance_has_exit: false,
      failed: false,
      closed: true,
    },
    approved_executable,
    consumed: false,
    cleanup_status: 0,
  }
}

///|

///|
/// Prepares a child whose executable authority is an already authenticated,
/// descriptor-pinned object. No path is accepted by this route.
pub fn prepare_child_from_approved_executable(
  executable : @executable_capability.ApprovedExecutableHandle,
  limits : ProcessLimits,
) -> PreparedChildProcess {
  prepare_approved_child(executable, limits)
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
pub fn PreparedChildProcess::spawn_with_approved_roots(
  self : PreparedChildProcess,
  roots : @inheritance.WorkerApprovedRootSpawnAuthority,
) -> ChildProcess raise ProcessError {
  self.require_fresh()
  self.consumed = true
  let status = raw_spawn_prepared_with_approved_executable_and_roots(
    self.child.handle,
    self.approved_executable,
    roots,
  )
  if status != 0 {
    ignore(raw_close(self.child.handle))
    raise SpawnFailed
  }
  self.child.closed = false
  self.child
}

///|
pub fn PreparedChildProcess::spawn_with_approved_roots_status(
  self : PreparedChildProcess,
  roots : @inheritance.WorkerApprovedRootSpawnAuthority,
) -> Int {
  if self.consumed {
    return 1
  }
  self.consumed = true
  let status = raw_spawn_prepared_with_approved_executable_and_roots(
    self.child.handle,
    self.approved_executable,
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
    self.maintenance_phase != 0 ||
    self.write_transfer_active ||
    self.write_source_conflicts_active_read(source) ||
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
    self.poison_transfers()
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
    self.poison_transfers()
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
    self.maintenance_phase != 0 ||
    self.read_transfer_active ||
    self.read_buffer_conflicts_active_write(destination) ||
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
    self.poison_transfers()
    return -(prefix + 1)
  }
  let byte_count = read_u32_prefix(self.read_prefix)
  if byte_count <= 0 ||
    byte_count > self.limits.max_frame_bytes ||
    byte_count > destination.length() {
    self.poison_transfers()
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
    self.poison_transfers()
    return -(payload + 1)
  }
  byte_count
}

///|
pub fn ChildProcess::startup_close_status(self : ChildProcess) -> Int {
  if self.closed {
    return 0
  }
  if self.maintenance_phase != 0 {
    return 1
  }
  self.clear_pending_transfers()
  let status = raw_close(self.handle)
  self.closed = raw_is_closed(self.handle)
  if status != 0 {
    self.failed = true
  }
  status
}
