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
fn ChildProcess::require_usable(
  self : ChildProcess,
  operation : ProcessOperation,
) -> Unit raise ProcessError {
  self.require_open(operation)
  if self.failed {
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
  {
    handle,
    limits,
    write_prefix: FixedArray::make(4, b'\x00'),
    read_prefix: FixedArray::make(4, b'\x00'),
    failed: false,
    closed: false,
  }
}

///|
/// Spawns one exact executable with model and kernel roots inherited at their
/// fixed child roles. The reusable pair remains owned by the caller and may be
/// used for a later replacement child.
pub fn spawn_with_approved_roots(
  executable : Bytes,
  roots : @inheritance.WorkerApprovedRoots,
  limits : ProcessLimits,
) -> ChildProcess raise ProcessError {
  let status = Ref(0)
  let handle = raw_spawn_with_approved_roots(executable, roots, status)
  if status.val != 0 {
    ignore(raw_close(handle))
    raise SpawnFailed
  }
  {
    handle,
    limits,
    write_prefix: FixedArray::make(4, b'\x00'),
    read_prefix: FixedArray::make(4, b'\x00'),
    failed: false,
    closed: false,
  }
}

///|
/// Opens the private socket inherited by a worker on stdin/stdout.
pub fn open_inherited(limits : ProcessLimits) -> InheritedChannel {
  {
    limits,
    write_prefix: FixedArray::make(4, b'\x00'),
    read_prefix: FixedArray::make(4, b'\x00'),
    failed: false,
  }
}

///|
/// Writes an exact fixed-buffer range or fails without reporting a short write.
fn ChildProcess::write_exact(
  self : ChildProcess,
  source : FixedArray[Byte],
  offset~ : Int,
  byte_count~ : Int,
  timeout_millis? : Int = self.limits.io_timeout_millis,
) -> Unit raise ProcessError {
  self.require_usable(Write)
  if offset < 0 ||
    byte_count < 0 ||
    timeout_millis <= 0 ||
    timeout_millis > 600000 ||
    byte_count > self.limits.max_frame_bytes ||
    offset > source.length() - byte_count {
    raise InvalidRange(Write)
  }
  raise_status(
    raw_write_exact(self.handle, source, offset, byte_count, timeout_millis),
    Write,
  ) catch {
    error => {
      self.failed = true
      raise error
    }
  }
}

///|
/// Reads exactly one fixed-buffer range; EOF before completion is an error.
fn ChildProcess::read_exact(
  self : ChildProcess,
  destination : FixedArray[Byte],
  offset~ : Int,
  byte_count~ : Int,
  timeout_millis? : Int = self.limits.io_timeout_millis,
) -> Unit raise ProcessError {
  self.require_usable(Read)
  if offset < 0 ||
    byte_count < 0 ||
    timeout_millis <= 0 ||
    timeout_millis > 600000 ||
    byte_count > self.limits.max_frame_bytes ||
    offset > destination.length() - byte_count {
    raise InvalidRange(Read)
  }
  raise_status(
    raw_read_exact(self.handle, destination, offset, byte_count, timeout_millis),
    Read,
  ) catch {
    error => {
      self.failed = true
      raise error
    }
  }
}

///|
fn write_u32_prefix(prefix : FixedArray[Byte], value : Int) -> Unit {
  prefix[0] = value.to_byte()
  prefix[1] = (value >> 8).to_byte()
  prefix[2] = (value >> 16).to_byte()
  prefix[3] = (value >> 24).to_byte()
}

///|
fn read_u32_prefix(prefix : FixedArray[Byte]) -> Int {
  prefix[0].to_int() |
  (prefix[1].to_int() << 8) |
  (prefix[2].to_int() << 16) |
  (prefix[3].to_int() << 24)
}

///|
/// Writes one length-delimited frame from caller-owned fixed storage.
pub fn ChildProcess::write_frame(
  self : ChildProcess,
  source : FixedArray[Byte],
  offset~ : Int,
  byte_count~ : Int,
) -> Unit raise ProcessError {
  self.write_frame_with_timeout(
    source,
    offset~,
    byte_count~,
    timeout_millis=self.limits.io_timeout_millis,
  )
}

///|
/// Writes one bounded frame with a caller-selected validated per-operation
/// timeout. The prefix and payload exact writes each use this bound.
pub fn ChildProcess::write_frame_with_timeout(
  self : ChildProcess,
  source : FixedArray[Byte],
  offset~ : Int,
  byte_count~ : Int,
  timeout_millis~ : Int,
) -> Unit raise ProcessError {
  self.require_usable(Write)
  if byte_count <= 0 ||
    byte_count > self.limits.max_frame_bytes ||
    timeout_millis <= 0 ||
    timeout_millis > 600000 ||
    offset < 0 ||
    offset > source.length() - byte_count {
    raise InvalidRange(Write)
  }
  write_u32_prefix(self.write_prefix, byte_count)
  self.write_exact(self.write_prefix, offset=0, byte_count=4, timeout_millis~)
  self.write_exact(source, offset~, byte_count~, timeout_millis~)
}

///|
/// Reads one complete bounded frame into caller-owned fixed storage.
pub fn ChildProcess::read_frame(
  self : ChildProcess,
  destination : FixedArray[Byte],
) -> Int raise ProcessError {
  self.read_frame_with_timeout(
    destination,
    timeout_millis=self.limits.io_timeout_millis,
  )
}

///|
/// Reads one bounded frame with a caller-selected validated per-operation
/// timeout. The prefix and payload exact reads each use this bound.
pub fn ChildProcess::read_frame_with_timeout(
  self : ChildProcess,
  destination : FixedArray[Byte],
  timeout_millis~ : Int,
) -> Int raise ProcessError {
  self.require_usable(Read)
  if timeout_millis <= 0 || timeout_millis > 600000 {
    raise InvalidRange(Read)
  }
  self.read_exact(self.read_prefix, offset=0, byte_count=4, timeout_millis~)
  let byte_count = read_u32_prefix(self.read_prefix)
  if byte_count <= 0 ||
    byte_count > self.limits.max_frame_bytes ||
    byte_count > destination.length() {
    self.failed = true
    raise InvalidRange(Read)
  }
  self.read_exact(destination, offset=0, byte_count~, timeout_millis~)
  byte_count
}

///|
fn InheritedChannel::require_usable(
  self : InheritedChannel,
  operation : ProcessOperation,
) -> Unit raise ProcessError {
  if self.failed {
    raise InvalidState(operation)
  }
}

///|
fn InheritedChannel::write_exact(
  self : InheritedChannel,
  source : FixedArray[Byte],
  offset : Int,
  byte_count : Int,
) -> Unit raise ProcessError {
  self.require_usable(Write)
  raise_status(
    raw_inherited_write_exact(
      source,
      offset,
      byte_count,
      self.limits.io_timeout_millis,
    ),
    Write,
  ) catch {
    error => {
      self.failed = true
      raise error
    }
  }
}

///|
fn InheritedChannel::read_exact(
  self : InheritedChannel,
  destination : FixedArray[Byte],
  offset : Int,
  byte_count : Int,
) -> Unit raise ProcessError {
  self.require_usable(Read)
  raise_status(
    raw_inherited_read_exact(
      destination,
      offset,
      byte_count,
      self.limits.io_timeout_millis,
    ),
    Read,
  ) catch {
    error => {
      self.failed = true
      raise error
    }
  }
}

///|
/// Writes one canonical length-delimited worker frame.
pub fn InheritedChannel::write_frame(
  self : InheritedChannel,
  source : FixedArray[Byte],
  offset~ : Int,
  byte_count~ : Int,
) -> Unit raise ProcessError {
  self.require_usable(Write)
  if byte_count <= 0 ||
    byte_count > self.limits.max_frame_bytes ||
    offset < 0 ||
    offset > source.length() - byte_count {
    raise InvalidRange(Write)
  }
  write_u32_prefix(self.write_prefix, byte_count)
  self.write_exact(self.write_prefix, 0, 4)
  self.write_exact(source, offset, byte_count)
}

///|
/// Reads one canonical length-delimited worker frame.
pub fn InheritedChannel::read_frame(
  self : InheritedChannel,
  destination : FixedArray[Byte],
) -> Int raise ProcessError {
  self.require_usable(Read)
  self.read_exact(self.read_prefix, 0, 4)
  let byte_count = read_u32_prefix(self.read_prefix)
  if byte_count <= 0 ||
    byte_count > self.limits.max_frame_bytes ||
    byte_count > destination.length() {
    self.failed = true
    raise InvalidRange(Read)
  }
  self.read_exact(destination, 0, byte_count)
  byte_count
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
