///|
priv struct StreamingIo {
  clock_output : Ref[UInt64]
  tx : FixedArray[Byte]
  rx : FixedArray[Byte]
  staging : FixedArray[Byte]
  mut phase : Int
  mut command : @streaming_ir.Command
}

///|
fn new_streaming_io() -> StreamingIo {
  {
    clock_output: Ref(0UL),
    tx: FixedArray::make(68, b'\x00'),
    rx: FixedArray::make(64, b'\x00'),
    staging: FixedArray::make(64, b'\x00'),
    phase: 0,
    command: {
      worker: 0UL,
      sequence: 0UL,
      opcode: Poll,
      page: 0,
      page_generation: 0,
      host: { slot: -1, generation: 0, },
    },
  }
}

///|
pub fn RootBoundWorkerProcessSupervisor::streaming_now_millis(
  self : RootBoundWorkerProcessSupervisor,
) -> UInt64 raise RootBoundWorkerProcessError {
  let output = self.process.streaming_io.clock_output
  if !@monotonic_clock.MonotonicClock::system().read_into(output) {
    raise RootBoundInvalidLifecycle
  }
  output.val
}

///|
pub fn RootBoundWorkerProcessSupervisor::streaming_exchange_pending(
  self : RootBoundWorkerProcessSupervisor,
) -> Bool {
  self.process.streaming_io.phase != 0
}

///|
pub fn RootBoundWorkerProcessSupervisor::begin_streaming_exchange(
  self : RootBoundWorkerProcessSupervisor,
  command : @streaming_ir.Command,
) -> Unit raise RootBoundWorkerProcessError {
  if !self.has_exchange_capacity() {
    raise RootBoundFailure(Process(Backpressured))
  }
  guard self.process.startup.streaming() is Some(capability) else {
    raise RootBoundInvalidLifecycle
  }
  if !command.valid(capability, self.process.startup.streaming_epoch()) {
    raise RootBoundInvalidLifecycle
  }
  self.transport_mode = ROOT_TRANSPORT_EXCHANGE
  let io = self.process.streaming_io
  @streaming_ir.encode({ command, status: Pending, host: command.host, }, io.rx) catch {
    _ => raise RootBoundInvalidLifecycle
  }
  // Native framed writes reserve four bytes before the immutable payload.
  for i in 0..<64 {
    io.tx[i + 4] = io.rx[i]
  }
  self.process
  .require_child()
  .begin_write_frame(
    io.tx,
    offset=4,
    byte_count=64,
    timeout_millis=self.limits.io_timeout_millis(),
  ) catch {
    error => self.fail_exchange(transport_error(error))
  }
  io.command = command
  io.phase = 1
}

///|
/// Cooperative transport progression; pending CUDA work uses further explicit
/// Poll commands and never blocks the service on a device wait.
pub fn RootBoundWorkerProcessSupervisor::progress_streaming_exchange(
  self : RootBoundWorkerProcessSupervisor,
) -> @streaming_ir.ReplyOption raise RootBoundWorkerProcessError {
  if self.lifecycle != RootBoundReady {
    raise RootBoundInvalidLifecycle
  }
  let io = self.process.streaming_io
  if io.phase == 1 {
    if !(self.process.require_child().progress_write_frame() catch {
        error => self.fail_exchange(transport_error(error))
      }) {
      return None
    }
    self.process
    .require_child()
    .begin_read_frame(
      io.rx,
      staging=io.staging,
      timeout_millis=self.limits.io_timeout_millis(),
    ) catch {
      error => self.fail_exchange(transport_error(error))
    }
    io.phase = 2
    return None
  }
  if io.phase != 2 {
    raise RootBoundInvalidLifecycle
  }
  let length = self.process.require_child().progress_read_frame() catch {
      error => self.fail_exchange(transport_error(error))
    }
  guard length is Some(count) else { return None }
  let reply = @streaming_ir.decode(io.rx, count) catch {
    _ => self.fail_exchange(Invalid(Completion))
  }
  if !reply.command.same_command(io.command) {
    self.fail_exchange(Invalid(Completion))
  }
  io.phase = 0
  Some(reply)
}
