///|
priv enum ClientState {
  Idle
  Sending(Int)
  Receiving(Int)
  InputsReady
  Retired
  FailedState
  Closed
}

///|
/// Coordinator-side control endpoint. It transports the canonical plan, then
/// run/commit/release commands. Only a retired terminal rank can return a token;
/// the coordinator still waits for every other rank before publishing it.
pub struct DecoderStageClient {
  priv channel : @channel.RemoteChannel
  priv message : FixedArray[Byte]
  priv outputs_token : Bool
  priv maximum_token : Int
  priv mut expect_token : Bool
  priv mut result_failed : Bool
  priv mut token : Int
  priv mut commit_failed : Bool
  priv mut epoch : UInt64
  priv mut state : ClientState
  priv mut busy : Bool
}

///|
pub fn DecoderStageClient::new(
  connection : @socket.Tcp,
  limits : @protocol.WorkerProtocolLimits,
  outputs_token~ : Bool,
  maximum_buffer_bytes~ : Int64,
  timeout_millis~ : UInt64,
) -> Self raise {
  guard maximum_buffer_bytes >= required_buffer_bytes(limits) else {
    raise StageControlError::InvalidConfiguration
  }
  let capacity = @wire.plan_frame_byte_capacity(limits) + control_header_bytes
  let message = FixedArray::make(capacity, b'\x00')
  let channel = @channel.RemoteChannel::new(
    connection,
    max_frame_bytes=capacity,
    timeout_millis~,
  )
  {
    channel,
    message,
    outputs_token,
    maximum_token: limits.max_token_id(),
    expect_token: false,
    result_failed: false,
    commit_failed: false,
    token: 0,
    epoch: 0UL,
    state: Idle,
    busy: false,
  }
}

///|
pub fn DecoderStageClient::prepare(
  self : Self,
  plan : @wire.ValidatedPlanFrame,
  epoch~ : UInt64,
  local_now_millis~ : UInt64,
) -> Unit raise {
  guard !self.busy && self.state is Idle && epoch > self.epoch else {
    raise StageControlError::InvalidState
  }
  let length = plan.length()
  guard length <= self.message.length() - control_header_bytes else {
    raise StageControlError::InvalidMessage
  }
  let expect_token = self.outputs_token &&
    (
      plan.decode_count() == 1 ||
      (plan.prefill_count() == 1 && plan.prefill_row(0).produces_token())
    )
  plan.copy_to(self.message, destination_offset=control_header_bytes)
  header(self.message, prepare_kind, epoch, 0, 0)
  self.state = FailedState
  self.channel.begin_send(
    self.message,
    length + control_header_bytes,
    local_now_millis,
  )
  self.epoch = epoch
  self.expect_token = expect_token
  self.state = Sending(inputs_kind)
}

///|
fn DecoderStageClient::command(
  self : Self,
  kind : Int,
  reply : Int,
  flags : Int,
  now : UInt64,
) -> Unit raise {
  header(self.message, kind, self.epoch, flags, 0)
  self.state = FailedState
  self.channel.begin_send(self.message, control_header_bytes, now)
  self.state = Sending(reply)
}

///|
pub fn DecoderStageClient::run(
  self : Self,
  local_now_millis~ : UInt64,
) -> Unit raise {
  guard !self.busy && self.state is InputsReady else {
    raise StageControlError::InvalidState
  }
  self.command(run_kind, result_kind, 0, local_now_millis)
}

///|
pub fn DecoderStageClient::commit(
  self : Self,
  pipeline_failed~ : Bool,
  local_now_millis~ : UInt64,
) -> Unit raise {
  guard !self.busy && self.state is Retired else {
    raise StageControlError::InvalidState
  }
  self.commit_failed = pipeline_failed || self.result_failed
  self.command(
    commit_kind,
    committed_kind,
    if self.commit_failed {
      1
    } else {
      0
    },
    local_now_millis,
  )
}

///|
pub fn DecoderStageClient::release_request(
  self : Self,
  local_now_millis~ : UInt64,
) -> Unit raise {
  guard !self.busy && self.state is Idle && self.epoch > 0UL else {
    raise StageControlError::InvalidState
  }
  self.command(release_kind, released_kind, 0, local_now_millis)
}

///|
fn DecoderStageClient::receipt(
  self : Self,
  kind : Int,
  length : Int,
) -> Unit raise {
  validate_header(self.message, length)
  guard length == control_header_bytes &&
    read_i32(self.message, 0) == kind &&
    read_u64(self.message, 8) == self.epoch else {
    raise StageControlError::InvalidMessage
  }
  let flags = read_i32(self.message, 4)
  let token = read_i32(self.message, 16)
  if kind == result_kind {
    guard flags == 1 || flags == (if self.expect_token { 2 } else { 0 }) else {
      raise StageControlError::InvalidMessage
    }
    if flags == 2 {
      guard token >= 0 && token <= self.maximum_token else {
        raise StageControlError::InvalidMessage
      }
    } else if token != 0 {
      raise StageControlError::InvalidMessage
    }
    self.result_failed = flags == 1
    self.token = token
    self.state = Retired
  } else if kind == committed_kind {
    guard flags == (if self.commit_failed { 1 } else { 0 }) && token == 0 else {
      raise StageControlError::InvalidMessage
    }
    self.state = if self.commit_failed { FailedState } else { Idle }
  } else {
    guard flags == 0 && token == 0 else {
      raise StageControlError::InvalidMessage
    }
    self.state = if kind == inputs_kind { InputsReady } else { Idle }
  }
}

///|
/// Returns true when the current prepare/run/commit/release exchange retires.
/// Errors retain FailedState; close the endpoint and drain its rank/edges.
pub async fn DecoderStageClient::poll(
  self : Self,
  local_now_millis~ : UInt64,
) -> Bool {
  guard !self.busy && self.state is (Sending(_) | Receiving(_)) else {
    raise StageControlError::InvalidState
  }
  self.busy = true
  defer {
    self.busy = false
  }
  errdefer {
    self.state = FailedState
  }
  match self.state {
    Sending(kind) => {
      if self.channel.poll_send(local_now_millis) {
        self.channel.begin_receive(local_now_millis)
        self.state = Receiving(kind)
      }
      false
    }
    Receiving(kind) =>
      if self.channel.poll_receive(self.message, local_now_millis)
        is Some(length) {
        self.receipt(kind, length)
        true
      } else {
        false
      }
    _ => raise StageControlError::InvalidState
  }
}

///|
pub fn DecoderStageClient::is_terminal(self : Self) -> Bool {
  self.state is (FailedState | Closed)
}

///|
pub fn DecoderStageClient::execution_failed(self : Self) -> Bool raise {
  guard self.state is Retired else { raise StageControlError::InvalidState }
  self.result_failed
}

///|
pub fn DecoderStageClient::outputs_token(self : Self) -> Bool {
  self.outputs_token
}

///|
pub fn DecoderStageClient::sampled_token(self : Self) -> Int raise {
  guard self.state is Retired && !self.result_failed && self.expect_token else {
    raise StageControlError::InvalidState
  }
  self.token
}

///|
pub fn DecoderStageClient::close(self : Self) -> Unit raise {
  guard !self.busy else { raise StageControlError::InvalidState }
  self.channel.close()
  self.state = Closed
}
