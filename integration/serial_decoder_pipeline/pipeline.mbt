///|
priv enum PipelineState {
  Ready
  Preparing
  Executing
  Committing
  Releasing
  Failed
  Closed
}

///|
/// Whole-step coordinator for one local ingress and one remote terminal stage.
/// All models use the same canonical plan, rank protocol and explicit activation
/// effects. This owner is independent of layer types, CUDA and checkpoint names.
/// It owns the supplied frame, ranks/client and activation effects after new().
pub struct TwoRankDecoderPipeline {
  priv frame : @frame.SerialDecoderFrame
  priv ingress : @rank.SerialDecoderRank
  priv remote : @control.DecoderStageClient
  priv local_completion : @wire.CompletionFrameBuffer
  priv begin_activation : (UInt64, Int, UInt64) -> Unit raise
  priv poll_activation : async (UInt64) -> Bool
  priv close_activation : () -> Unit raise
  priv timeout_millis : UInt64
  priv mut last_tick : UInt64
  priv mut deadline : UInt64
  priv mut epoch : UInt64
  priv mut local_ready : Bool
  priv mut remote_ready : Bool
  priv mut activation_done : Bool
  priv mut step_failed : Bool
  priv mut token : Int
  priv mut busy : Bool
  priv mut state : PipelineState
}

///|
pub fn TwoRankDecoderPipeline::new(
  frame : @frame.SerialDecoderFrame,
  ingress : @rank.SerialDecoderRank,
  remote : @control.DecoderStageClient,
  limits : @protocol.WorkerProtocolLimits,
  inference_limits : @inference.InferenceLimits,
  begin_activation~ : (UInt64, Int, UInt64) -> Unit raise,
  poll_activation~ : async (UInt64) -> Bool,
  close_activation~ : () -> Unit raise,
  timeout_millis? : UInt64 = 300000UL,
) -> Self raise {
  guard !ingress.outputs_token() &&
    remote.outputs_token() &&
    frame.geometry() == ingress.ports().geometry() &&
    timeout_millis > 0UL &&
    timeout_millis <= 300000UL else {
    raise @frame.SerialDecoderError::InvalidCapacity
  }
  {
    frame,
    ingress,
    remote,
    begin_activation,
    poll_activation,
    close_activation,
    timeout_millis,
    last_tick: 0UL,
    deadline: 0UL,
    local_completion: @wire.CompletionFrameBuffer::new(limits, inference_limits),
    epoch: 0UL,
    local_ready: false,
    remote_ready: false,
    activation_done: false,
    step_failed: false,
    token: 0,
    busy: false,
    state: Ready,
  }
}

///|
/// The same plan remains immutable until completion. Each rank has independent
/// retained history and a local writer lease; only this frame publishes output.
pub fn TwoRankDecoderPipeline::begin(
  self : Self,
  plan : @wire.ValidatedPlanFrame,
  writer : @wire.CompletionFrameWriter,
  epoch~ : UInt64,
  local_now_millis~ : UInt64,
) -> Unit raise {
  guard !self.busy &&
    self.state is Ready &&
    epoch > self.epoch &&
    local_now_millis >= self.last_tick &&
    local_now_millis <= 0xffffffffffffffffUL - self.timeout_millis else {
    raise @frame.SerialDecoderError::Busy
  }
  self.frame.stage(plan, writer)
  self.state = Failed
  self.ingress.begin(plan, self.local_completion.writer_for(plan))
  self.remote.prepare(plan, epoch~, local_now_millis~)
  self.epoch = epoch
  self.last_tick = local_now_millis
  self.deadline = local_now_millis + self.timeout_millis
  self.local_ready = false
  self.remote_ready = false
  self.activation_done = false
  self.state = Preparing
}

///|
/// One cooperative phase. Local retirement, residual transfer and remote
/// execution overlap, but publication waits for all three and commit receipt.
pub async fn TwoRankDecoderPipeline::progress(
  self : Self,
  local_now_millis~ : UInt64,
) -> Bool {
  guard !self.busy &&
    self.state is (Preparing | Executing | Committing | Releasing) else {
    raise @frame.SerialDecoderError::Busy
  }
  self.busy = true
  defer {
    self.busy = false
  }
  errdefer {
    self.state = Failed
  }
  guard local_now_millis >= self.last_tick && local_now_millis < self.deadline else {
    raise @control.StageControlError::Failed
  }
  self.last_tick = local_now_millis
  match self.state {
    Preparing => {
      if !self.local_ready {
        self.local_ready = self.ingress.inputs_ready()
      }
      if !self.remote_ready {
        self.remote_ready = self.remote.poll(local_now_millis~)
      }
      if self.local_ready && self.remote_ready {
        self.ingress.submit()
        (self.begin_activation)(
          self.epoch,
          self.frame.live_rows(),
          local_now_millis,
        )
        self.remote.run(local_now_millis~)
        self.local_ready = false
        self.remote_ready = false
        self.state = Executing
      }
      false
    }
    Executing => {
      if !self.local_ready {
        self.local_ready = self.ingress.progress()
      }
      if !self.activation_done {
        self.activation_done = (self.poll_activation)(local_now_millis)
      }
      if !self.remote_ready {
        self.remote_ready = self.remote.poll(local_now_millis~)
      }
      if self.local_ready && self.activation_done && self.remote_ready {
        self.step_failed = self.ingress.execution_failed() ||
          self.remote.execution_failed()
        self.token = if !self.step_failed && self.frame.produces_token() {
          self.remote.sampled_token()
        } else {
          0
        }
        self.remote.commit(pipeline_failed=self.step_failed, local_now_millis~)
        self.state = Committing
      }
      false
    }
    Committing => {
      if !self.remote.poll(local_now_millis~) {
        return false
      }
      self.ingress.commit(pipeline_failed=self.step_failed)
      ignore(
        self.frame.complete(
          sampled_token=self.token,
          execution_failed=self.step_failed,
        ),
      )
      self.state = if self.frame.is_faulted() { Failed } else { Ready }
      true
    }
    Releasing =>
      if self.remote.poll(local_now_millis~) {
        self.frame.release_request()
        self.state = Ready
        true
      } else {
        false
      }
    _ => raise @frame.SerialDecoderError::Busy
  }
}

///|
pub fn TwoRankDecoderPipeline::completion(
  self : Self,
) -> @wire.ValidatedCompletionFrame raise {
  guard self.state is (Ready | Failed) else {
    raise @frame.SerialDecoderError::Busy
  }
  self.frame.completion()
}

///|
/// Begin request retirement on both ranks. progress() must retire its receipt
/// before a different request may be staged and reset the model's histories.
pub fn TwoRankDecoderPipeline::release_request(
  self : Self,
  local_now_millis~ : UInt64,
) -> Unit raise {
  guard !self.busy &&
    self.state is Ready &&
    local_now_millis >= self.last_tick &&
    local_now_millis <= 0xffffffffffffffffUL - self.timeout_millis else {
    raise @frame.SerialDecoderError::Busy
  }
  self.state = Failed
  self.ingress.release_request()
  self.remote.release_request(local_now_millis~)
  self.last_tick = local_now_millis
  self.deadline = local_now_millis + self.timeout_millis
  self.state = Releasing
}

///|
/// Close networking, drain activation DMA, then local execution/ports. The
/// remote server separately drains its owned rank after control disconnect.
pub fn TwoRankDecoderPipeline::close(self : Self) -> Unit raise {
  guard !self.busy else { raise @frame.SerialDecoderError::Busy }
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.remote.close()
  (self.close_activation)()
  self.ingress.close()
  self.frame.close()
  self.state = Closed
}
