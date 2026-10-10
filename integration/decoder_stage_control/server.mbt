///|
priv enum ServerPhase {
  Idle
  Released
  Prepared
  Retired
  Poisoned
}

///|
priv enum ServerState {
  Receiving
  Uploading
  WaitingActivation
  Executing
  CommittedEffect
  Sending
  Terminal
  FailedState
  Closed
}

///|
/// Plain control transport for one prepared rank. Activation DMA is polled by
/// its own owner; ready_to_submit cannot report true until that owner retires.
/// The server owns the connection but borrows the rank. Close transport/edges
/// before closing rank resources. Cancellation never implies successful commit.
pub struct DecoderStageServer {
  priv rank : @rank.SerialDecoderRank
  priv channel : @channel.RemoteChannel
  priv message : FixedArray[Byte]
  priv plan_bytes : FixedArray[Byte]
  priv plan : @wire.PlanFrameBuffer
  priv completion : @wire.CompletionFrameBuffer
  priv ready_to_submit : () -> Bool raise
  priv arm_activation : (UInt64, Int, UInt64) -> Unit raise
  priv committed_effect : ((Bool) -> Unit raise)?
  priv poll_committed_effect : () -> Bool raise
  priv timeout_millis : UInt64
  priv mut last_tick : UInt64
  priv mut phase_deadline : UInt64
  priv mut epoch : UInt64
  priv mut phase : ServerPhase
  priv mut state : ServerState
  priv mut busy : Bool
  priv mut decode_step : Bool
}

///|
pub fn DecoderStageServer::new(
  rank : @rank.SerialDecoderRank,
  connection : @socket.Tcp,
  limits : @protocol.WorkerProtocolLimits,
  inference_limits : @inference.InferenceLimits,
  maximum_buffer_bytes~ : Int64,
  timeout_millis~ : UInt64,
  local_now_millis~ : UInt64,
  ready_to_submit~ : () -> Bool raise,
  arm_activation? : (UInt64, Int, UInt64) -> Unit raise = (_, _, _) => (),
  committed_effect? : (Bool) -> Unit raise,
  poll_committed_effect? : () -> Bool raise = () => true,
) -> Self raise {
  guard maximum_buffer_bytes >= required_buffer_bytes(limits) &&
    timeout_millis > 0UL &&
    timeout_millis <= 300000UL &&
    local_now_millis <= 0xffffffffffffffffUL - timeout_millis else {
    raise StageControlError::InvalidConfiguration
  }
  let plan_capacity = @wire.plan_frame_byte_capacity(limits)
  let plan = @wire.PlanFrameBuffer::new(limits, inference_limits)
  let completion = @wire.CompletionFrameBuffer::new(limits, inference_limits)
  let message = FixedArray::make(plan_capacity + control_header_bytes, b'\x00')
  let plan_bytes = FixedArray::make(plan_capacity, b'\x00')
  let channel = @channel.RemoteChannel::new(
    connection,
    max_frame_bytes=message.length(),
    timeout_millis~,
  )
  channel.begin_receive(local_now_millis, idle=true)
  {
    rank,
    channel,
    message,
    plan_bytes,
    plan,
    completion,
    ready_to_submit,
    arm_activation,
    committed_effect,
    poll_committed_effect,
    timeout_millis,
    last_tick: local_now_millis,
    phase_deadline: local_now_millis + timeout_millis,
    epoch: 0UL,
    phase: Idle,
    state: Receiving,
    busy: false,
    decode_step: false,
  }
}

///|
fn DecoderStageServer::reply(
  self : Self,
  kind : Int,
  flags : Int,
  token : Int,
  now : UInt64,
) -> Unit raise {
  header(self.message, kind, self.epoch, flags, token)
  self.channel.begin_send(self.message, control_header_bytes, now)
  self.state = Sending
}

///|
fn DecoderStageServer::command(
  self : Self,
  length : Int,
  now : UInt64,
) -> Unit raise {
  validate_header(self.message, length)
  let kind = read_i32(self.message, 0)
  let flags = read_i32(self.message, 4)
  let epoch = read_u64(self.message, 8)
  guard read_i32(self.message, 16) == 0 else {
    raise StageControlError::InvalidMessage
  }
  if kind == prepare_kind {
    guard self.phase is (Idle | Released) &&
      flags == 0 &&
      epoch > self.epoch &&
      length > control_header_bytes else {
      raise StageControlError::InvalidMessage
    }
    let bytes = length - control_header_bytes
    for i in 0..<bytes {
      self.plan_bytes[i] = self.message[control_header_bytes + i]
    }
    let plan = self.plan.load(self.plan_bytes, bytes)
    self.decode_step = plan.decode_count() > 0
    let writer = self.completion.writer_for(plan)
    self.rank.begin(plan, writer)
    self.epoch = epoch
    self.phase_deadline = now + self.timeout_millis
    self.state = Uploading
    return
  }
  guard length == control_header_bytes && epoch == self.epoch && epoch > 0UL else {
    raise StageControlError::InvalidMessage
  }
  if kind == run_kind && flags == 0 && self.phase is Prepared {
    self.phase_deadline = now + self.timeout_millis
    self.state = WaitingActivation
  } else if kind == commit_kind &&
    (flags == 0 || flags == 1) &&
    self.phase is Retired {
    let failed = flags == 1 || self.rank.execution_failed()
    self.rank.commit(pipeline_failed=failed)
    self.phase = if failed { Poisoned } else { Idle }
    if !failed && self.committed_effect is Some(begin_effect) {
      begin_effect(self.decode_step)
      self.phase_deadline = now + self.timeout_millis
      self.state = CommittedEffect
    } else {
      self.reply(committed_kind, if failed { 1 } else { 0 }, 0, now)
    }
  } else if kind == release_kind && flags == 0 && self.phase is Idle {
    self.rank.release_request()
    self.phase = Released
    self.reply(released_kind, 0, 0, now)
  } else {
    raise StageControlError::InvalidMessage
  }
}

///|
/// One bounded transport or rank phase. The activation owner remains separate;
/// waiting for it never submits early or blocks this cooperative event loop.
pub async fn DecoderStageServer::poll(
  self : Self,
  local_now_millis~ : UInt64,
) -> Unit {
  guard !self.busy else { raise StageControlError::InvalidState }
  guard !(self.state is (FailedState | Closed)) else {
    raise StageControlError::Failed
  }
  self.busy = true
  defer {
    self.busy = false
  }
  errdefer {
    self.state = FailedState
  }
  guard local_now_millis >= self.last_tick &&
    local_now_millis <= 0xffffffffffffffffUL - self.timeout_millis else {
    raise @channel.RemoteChannelError::ClockRegressed
  }
  self.last_tick = local_now_millis
  if self.state is (Uploading | WaitingActivation | Executing | CommittedEffect) &&
    local_now_millis >= self.phase_deadline {
    raise @channel.RemoteChannelError::TimedOut
  }
  match self.state {
    Receiving =>
      match
        self.channel.poll_receive(
          self.message,
          local_now_millis,
          allow_idle_eof=self.phase is Released,
        ) {
        Some(length) => self.command(length, local_now_millis)
        None => if !self.channel.is_open() { self.state = Terminal }
      }
    Uploading =>
      if self.rank.inputs_ready() {
        (self.arm_activation)(
          self.epoch,
          self.rank.live_rows(),
          local_now_millis,
        )
        self.phase = Prepared
        self.reply(inputs_kind, 0, 0, local_now_millis)
      }
    WaitingActivation =>
      if (self.ready_to_submit)() {
        self.rank.submit()
        self.state = Executing
      }
    Executing =>
      if self.rank.progress() {
        let failed = self.rank.execution_failed()
        let output = !failed && self.rank.produces_token()
        let token = if output { self.rank.sampled_token() } else { 0 }
        self.phase = Retired
        self.reply(
          result_kind,
          (if failed { 1 } else { 0 }) | (if output { 2 } else { 0 }),
          token,
          local_now_millis,
        )
      }
    CommittedEffect =>
      if (self.poll_committed_effect)() {
        self.reply(committed_kind, 0, 0, local_now_millis)
      }
    Sending =>
      if self.channel.poll_send(local_now_millis) {
        if self.phase is Poisoned {
          self.state = Terminal
        } else {
          self.channel.begin_receive(
            local_now_millis,
            idle=self.phase is (Idle | Released),
          )
          self.state = Receiving
        }
      }
    Terminal => ()
    _ => raise StageControlError::Failed
  }
}

///|
pub fn DecoderStageServer::is_terminal(self : Self) -> Bool {
  self.state is (Terminal | FailedState | Closed)
}

///|
pub fn DecoderStageServer::close(self : Self) -> Unit raise {
  guard !self.busy else { raise StageControlError::InvalidState }
  self.channel.close()
  self.state = Closed
}
