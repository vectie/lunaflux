///|
/// Stateful batched workers need an explicit release acknowledgment. Older
/// page-only workers have no request-keyed recurrent ownership to release.
pub fn RootBoundWorkerProcessSupervisor::requires_request_retirement(
  self : Self,
) -> Bool {
  self.process.source.recipe() == PackedGlmBatchedTensorV14
}

///|
/// One nonblocking transport transition. No slot is reusable while this
/// receipt is pending; neither the plan sequence nor model generation changes.
pub fn RootBoundWorkerProcessSupervisor::progress_request_retirement(
  self : Self,
  request_id : UInt64,
  generation : UInt64,
) -> Bool raise RootBoundWorkerProcessError {
  guard self.lifecycle == RootBoundReady && self.requires_request_retirement() else {
    raise RootBoundInvalidLifecycle
  }
  let io = self.process.streaming_io
  let command : @worker_wire.RequestRetirement = {
    predecessor: self.process.last_retired,
    request_id,
    generation,
    acknowledged: false,
  }
  if io.phase == 0 {
    if !self.has_exchange_capacity() {
      return false
    }
    command.encode(io.rx) catch {
      _ => raise RootBoundInvalidLifecycle
    }
    for i in 0..<40 {
      io.tx[i + 4] = io.rx[i]
    }
    self.process
    .require_child()
    .begin_write_frame(
      io.tx,
      offset=4,
      byte_count=40,
      timeout_millis=self.limits.io_timeout_millis(),
    ) catch {
      e => self.fail_exchange(transport_error(e))
    }
    io.retirement = command
    io.phase = 3
    return false
  }
  guard io.phase is (3 | 4) && io.retirement == command else {
    raise RootBoundInvalidLifecycle
  }
  if io.phase == 3 {
    if !(self.process.require_child().progress_write_frame() catch {
        e => self.fail_exchange(transport_error(e))
      }) {
      return false
    }
    self.process
    .require_child()
    .begin_read_frame(
      io.rx,
      staging=io.staging,
      timeout_millis=self.limits.io_timeout_millis(),
    ) catch {
      e => self.fail_exchange(transport_error(e))
    }
    io.phase = 4
    return false
  }
  let length = self.process.require_child().progress_read_frame() catch {
      e => self.fail_exchange(transport_error(e))
    }
  guard length is Some(count) else { return false }
  let response = @worker_wire.decode_request_retirement(io.rx, count) catch {
    _ => self.fail_exchange(Invalid(Completion))
  }
  guard response == { ..command, acknowledged: true, } else {
    self.fail_exchange(Invalid(Completion))
  }
  io.phase = 0
  true
}
