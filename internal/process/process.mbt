///|
fn raise_status(
  status : Int,
  operation : ProcessOperation,
) -> Unit raise ProcessError {
  match status {
    0 => ()
    1 => raise InvalidState(operation)
    2 => raise Timeout(operation)
    3 => raise ChannelClosed(operation)
    _ => raise OperationFailed(operation)
  }
}

///|
fn ChildProcess::require_open(
  self : ChildProcess,
  operation : ProcessOperation,
) -> Unit raise ProcessError {
  if self.closed {
    raise InvalidState(operation)
  }
}

///|
/// Spawns one exact executable with an inherited private full-duplex channel.
pub fn spawn(
  executable : Bytes,
  limits : ProcessLimits,
) -> ChildProcess raise ProcessError {
  let status = Ref(0)
  let handle = raw_spawn(executable, status)
  if status.val != 0 {
    ignore(raw_close(handle))
    raise SpawnFailed
  }
  { handle, limits, closed: false }
}

///|
/// Writes an exact fixed-buffer range or fails without reporting a short write.
pub fn ChildProcess::write_exact(
  self : ChildProcess,
  source : FixedArray[Byte],
  offset~ : Int,
  byte_count~ : Int,
) -> Unit raise ProcessError {
  self.require_open(Write)
  if offset < 0 ||
    byte_count < 0 ||
    byte_count > self.limits.max_frame_bytes ||
    offset > source.length() - byte_count {
    raise InvalidRange(Write)
  }
  raise_status(
    raw_write_exact(
      self.handle,
      source,
      offset,
      byte_count,
      self.limits.io_timeout_millis,
    ),
    Write,
  )
}

///|
/// Reads exactly one fixed-buffer range; EOF before completion is an error.
pub fn ChildProcess::read_exact(
  self : ChildProcess,
  destination : FixedArray[Byte],
  offset~ : Int,
  byte_count~ : Int,
) -> Unit raise ProcessError {
  self.require_open(Read)
  if offset < 0 ||
    byte_count < 0 ||
    byte_count > self.limits.max_frame_bytes ||
    offset > destination.length() - byte_count {
    raise InvalidRange(Read)
  }
  raise_status(
    raw_read_exact(
      self.handle,
      destination,
      offset,
      byte_count,
      self.limits.io_timeout_millis,
    ),
    Read,
  )
}

///|
/// Publishes EOF to the child while retaining the read side for drain.
pub fn ChildProcess::shutdown_write(
  self : ChildProcess,
) -> Unit raise ProcessError {
  self.require_open(ShutdownWrite)
  raise_status(raw_shutdown_write(self.handle), ShutdownWrite)
}

///|
/// Waits for exact child termination with a bounded monotonic timeout.
pub fn ChildProcess::wait(self : ChildProcess) -> ChildExit raise ProcessError {
  self.require_open(Wait)
  let kind = Ref(0)
  let code = Ref(0)
  raise_status(
    raw_wait(self.handle, self.limits.shutdown_timeout_millis, kind, code),
    Wait,
  )
  { kind: kind.val, code: code.val }
}

///|
/// Requests graceful termination without consuming close authority.
pub fn ChildProcess::terminate(self : ChildProcess) -> Unit raise ProcessError {
  self.require_open(Terminate)
  raise_status(raw_terminate(self.handle), Terminate)
}

///|
/// Closes the channel, kills a live child, and reaps it exactly once.
pub fn ChildProcess::close(self : ChildProcess) -> Unit raise ProcessError {
  if self.closed {
    return
  }
  raise_status(raw_close(self.handle), Close)
  self.closed = true
}
