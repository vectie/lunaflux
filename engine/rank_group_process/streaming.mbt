///|
priv struct StreamingGroup {
  capability : @streaming_ir.Capability
  clock_output : Ref[UInt64]
  payload : FixedArray[Byte]
  mut control : @streaming_ir.Control
  mut transfer : @streaming_ir.GroupTransfer
  mut command : @streaming_ir.Command
  mut exchanging : Bool
  mut rank : Int
}

///|
fn new_streaming_group(
  capability : @streaming_ir.Capability,
  epoch : UInt64,
  ranks : Int,
) -> StreamingGroup {
  let command : @streaming_ir.Command = {
    worker: epoch,
    sequence: 0UL,
    opcode: Poll,
    page: 0,
    page_generation: 0,
    host: { slot: -1, generation: 0, },
  }
  {
    capability,
    clock_output: Ref(0UL),
    payload: FixedArray::make(64, b'\x00'),
    control: @streaming_ir.Control::new(epoch),
    transfer: try! @streaming_ir.GroupTransfer::new(command, ranks),
    command,
    exchanging: false,
    rank: 0,
  }
}

///|
pub fn RankGroupProcessSupervisor::streaming_now_millis(
  self : RankGroupProcessSupervisor,
) -> UInt64 raise RankGroupProcessError {
  guard self.streaming is Some(streaming) else { self.reject(Lifecycle) }
  if !self.clock.read_into(streaming.clock_output) {
    self.fail(Clock)
  }
  streaming.clock_output.val
}

///|
pub fn RankGroupProcessSupervisor::streaming_exchange_pending(
  self : RankGroupProcessSupervisor,
) -> Bool {
  self.streaming is Some(streaming) && streaming.exchanging
}

///|
pub fn RankGroupProcessSupervisor::streaming_busy(
  self : RankGroupProcessSupervisor,
) -> Bool {
  self.streaming is Some(streaming) &&
  (
    streaming.exchanging ||
    (
      streaming.transfer.command.sequence != 0UL &&
      streaming.transfer.remaining != 0
    )
  )
}

///|
pub fn RankGroupProcessSupervisor::begin_streaming_exchange(
  self : RankGroupProcessSupervisor,
  command : @streaming_ir.Command,
) -> Unit raise RankGroupProcessError {
  self.require_lifecycle(Ready)
  guard self.streaming is Some(streaming) &&
    !streaming.exchanging &&
    self.io_phase == 0 else {
    self.reject(Lifecycle)
  }
  let decision = streaming.control.decide(command, streaming.capability)
  if decision.action is Reject {
    self.reject(Binding)
  }
  // Spills are maintenance transactions. The scheduler never cancels them;
  // abandoning one requires whole-group recovery and reap.
  if command.opcode is Cancel && streaming.transfer.command.opcode is Spill {
    self.reject(Binding)
  }
  match command.opcode {
    Spill | Restore | Evict =>
      streaming.transfer = @streaming_ir.GroupTransfer::new(
        command,
        self.world_size(),
      ) catch {
        _ => self.reject(Binding)
      }
    Cancel =>
      streaming.transfer = streaming.transfer.cancel() catch {
        _ => self.reject(Binding)
      }
    Poll => ()
  }
  streaming.control = decision.next
  streaming.command = command
  streaming.rank = 0
  streaming.exchanging = true
}

///|
pub fn RankGroupProcessSupervisor::progress_streaming_exchange(
  self : RankGroupProcessSupervisor,
) -> @streaming_ir.ReplyOption raise RankGroupProcessError {
  self.require_lifecycle(Ready)
  guard self.streaming is Some(streaming) && streaming.exchanging else {
    self.reject(Lifecycle)
  }
  let rank = streaming.rank
  if rank == self.world_size() {
    streaming.exchanging = false
    if streaming.transfer.remaining == 0 {
      streaming.control = streaming.control.completed()
    }
    return Some(streaming.transfer.reply(streaming.command))
  }
  if streaming.transfer.rank_done(rank) {
    streaming.rank += 1
    return None
  }
  if self.io_phase == 0 {
    @streaming_ir.encode(
      {
        command: streaming.command,
        status: Pending,
        host: streaming.command.host,
      },
      streaming.payload,
    ) catch {
      _ => self.reject(Binding)
    }
    let length = @rank_group_wire.encode_rank_group_wire_payload_into(
      self.tx,
      self.wire_limits,
      self.binding(rank),
      StreamingRequest,
      ParentToChild,
      None,
      streaming.command.sequence,
      streaming.payload,
      64,
    ) catch {
      error => self.fail(Wire(error))
    }
    ignore(
      self.wire_transcript.accept_bytes(self.tx, length, self.wire_limits) catch {
        error => self.fail(Wire(error))
      },
    )
    self.io_rank = rank
    self.children[rank].begin_write(
      self.tx,
      length,
      self.limits.io_timeout_millis,
    ) catch {
      _ => self.terminal_transport_lost(rank)
    }
    self.io_phase = 1
    return None
  }
  if self.io_phase == 1 {
    if !(self.children[rank].progress_write() catch {
        _ => self.terminal_transport_lost(rank)
      }) {
      return None
    }
    self.children[rank].begin_read(
      self.rx,
      self.rx_staging,
      self.limits.io_timeout_millis,
    ) catch {
      _ => self.terminal_transport_lost(rank)
    }
    self.io_phase = 2
    return None
  }
  let received = self.children[rank].progress_read() catch {
    _ => self.terminal_transport_lost(rank)
  }
  guard received is Some(length) else { return None }
  let header = self.wire_transcript.accept_bytes(
    self.rx,
    length,
    self.wire_limits,
  ) catch {
    error => self.fail(Wire(error))
  }
  if header.kind() != StreamingReply || header.payload_length() != 64 {
    self.fail(Streaming(rank~))
  }
  let offset = @rank_group_wire.rank_group_wire_header_bytes()
  for index in 0..<64 {
    streaming.payload[index] = self.rx[offset + index]
  }
  let reply = @streaming_ir.decode(streaming.payload, 64) catch {
    _ => self.fail(Streaming(rank~))
  }
  if !reply.command.same_command(streaming.command) {
    self.fail(Streaming(rank~))
  }
  streaming.transfer = streaming.transfer.accept(rank, reply) catch {
    _ => self.fail(Streaming(rank~))
  }
  self.io_phase = 0
  self.io_rank = -1
  streaming.rank += 1
  None
}

///|
pub fn RankGroupProcessSupervisor::fail_streaming(
  self : RankGroupProcessSupervisor,
) -> Unit {
  if self.streaming is Some(_) && self.lifecycle_value == Ready {
    self.record_failure(Streaming(rank=-1))
  }
}
