///|
priv enum State {
  Ready
  Preparing
  Dispatching
  Executing
  Committing
  Releasing
  Failed
  Closed
}

///|
/// Whole-step TP owner. Both ranks receive the same immutable canonical plan;
/// model collectives are already bound in their executors. This owner publishes
/// only after both executions and the remote commit receipt retire.
pub struct BatchedTensorPipeline {
  priv frame : @frame.BatchedDecoderFrame
  priv local : @rank.BatchedDecoderRank
  priv remote : @control.DecoderStageClient
  priv local_completion : @wire.CompletionFrameBuffer
  priv samples : Array[Int]
  priv timeout : UInt64
  priv mut epoch : UInt64
  priv mut last_tick : UInt64
  priv mut deadline : UInt64
  priv mut local_ready : Bool
  priv mut remote_ready : Bool
  priv mut failed : Bool
  priv mut busy : Bool
  priv mut release_id : UInt64
  priv mut release_generation : UInt64
  priv mut state : State
}

///|
pub fn BatchedTensorPipeline::new(
  frame : @frame.BatchedDecoderFrame,
  local : @rank.BatchedDecoderRank,
  remote : @control.DecoderStageClient,
  limits : @protocol.WorkerProtocolLimits,
  inference_limits : @inference.InferenceLimits,
  timeout_millis? : UInt64 = 300000UL,
) -> Self raise {
  guard !local.outputs_token() &&
    remote.outputs_token() &&
    remote.is_batched() &&
    frame.geometry() == local.geometry() &&
    remote.output_capacity() >= frame.geometry().1 &&
    timeout_millis > 0UL &&
    timeout_millis <= 300000UL else {
    raise @frame.BatchFrameError::InvalidCapacity
  }
  {
    frame,
    local,
    remote,
    local_completion: @wire.CompletionFrameBuffer::new(limits, inference_limits),
    samples: Array::make(frame.geometry().1, 0),
    timeout: timeout_millis,
    epoch: 0UL,
    last_tick: 0UL,
    deadline: 0UL,
    local_ready: false,
    remote_ready: false,
    failed: false,
    busy: false,
    release_id: 0UL,
    release_generation: 0UL,
    state: Ready,
  }
}

///|
pub fn BatchedTensorPipeline::begin(
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
    local_now_millis <= 0xffffffffffffffffUL - self.timeout else {
    raise @frame.BatchFrameError::Busy
  }
  self.frame.stage(plan, writer)
  self.state = Failed
  self.local.begin(plan, self.local_completion.writer_for(plan))
  self.remote.prepare(plan, epoch~, local_now_millis~)
  self.epoch = epoch
  self.last_tick = local_now_millis
  self.deadline = local_now_millis + self.timeout
  self.local_ready = false
  self.remote_ready = false
  self.state = Preparing
}

///|
pub async fn BatchedTensorPipeline::progress(
  self : Self,
  local_now_millis~ : UInt64,
) -> Bool {
  guard !self.busy &&
    self.state is (Preparing | Dispatching | Executing | Committing | Releasing) else {
    raise @frame.BatchFrameError::Busy
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
        self.local_ready = self.local.inputs_ready()
      }
      if !self.remote_ready {
        self.remote_ready = self.remote.poll(local_now_millis~)
      }
      if self.local_ready && self.remote_ready {
        self.remote.run(local_now_millis~)
        self.local_ready = false
        self.remote_ready = false
        self.state = Dispatching
        // A long local graph enqueue must not delay the peer's run command.
        // Only the bounded send retires here; neither rank may publish yet.
        if self.remote.dispatch_run(local_now_millis~) {
          self.local.submit()
          self.state = Executing
        }
      }
      false
    }
    Dispatching => {
      if self.remote.dispatch_run(local_now_millis~) {
        self.local.submit()
        self.state = Executing
      }
      false
    }
    Executing => {
      if !self.local_ready {
        self.local_ready = self.local.progress()
      }
      if !self.remote_ready {
        self.remote_ready = self.remote.poll(local_now_millis~)
      }
      if self.local_ready && self.remote_ready {
        self.failed = self.local.execution_failed() ||
          self.remote.execution_failed()
        if !self.failed {
          guard self.remote.output_rows() == self.frame.output_rows() else {
            raise @control.StageControlError::InvalidMessage
          }
          for row in 0..<self.frame.output_rows() {
            self.samples[row] = self.remote.sampled_token_at(row)
          }
        }
        self.remote.commit(pipeline_failed=self.failed, local_now_millis~)
        self.state = Committing
      }
      false
    }
    Committing => {
      if !self.remote.poll(local_now_millis~) {
        return false
      }
      self.local.commit(pipeline_failed=self.failed)
      ignore(
        self.frame.complete(
          self.samples[:self.frame.output_rows()],
          execution_failed=self.failed,
        ),
      )
      self.state = if self.failed { Failed } else { Ready }
      true
    }
    Releasing => {
      if !self.remote.poll(local_now_millis~) {
        return false
      }
      self.local.retire_request(self.release_id, self.release_generation)
      self.frame.retire_request(self.release_id, self.release_generation)
      self.state = Ready
      true
    }
    _ => raise @frame.BatchFrameError::Busy
  }
}

///|
pub fn BatchedTensorPipeline::completion(
  self : Self,
) -> @wire.ValidatedCompletionFrame raise {
  guard !self.busy && self.state is (Ready | Failed) else {
    raise @frame.BatchFrameError::Busy
  }
  self.frame.completion()
}

///|
/// The scheduler first retires the in-flight batch (discarding cancelled
/// responses), then releases each cancelled/completed request. No cancellation
/// resets another request's KV or recurrent state and none is a device rollback.
pub fn BatchedTensorPipeline::release_request(
  self : Self,
  request_id : UInt64,
  generation : UInt64,
  local_now_millis~ : UInt64,
) -> Unit raise {
  guard !self.busy &&
    self.state is Ready &&
    local_now_millis >= self.last_tick &&
    local_now_millis <= 0xffffffffffffffffUL - self.timeout else {
    raise @frame.BatchFrameError::Busy
  }
  // Validate identity before either rank changes its retained ownership.
  ignore(self.frame.contains_request(request_id, generation))
  self.state = Failed
  self.remote.release_one(request_id, generation, local_now_millis~)
  self.release_id = request_id
  self.release_generation = generation
  self.last_tick = local_now_millis
  self.deadline = local_now_millis + self.timeout
  self.state = Releasing
}

///|
pub fn BatchedTensorPipeline::close(self : Self) -> Unit raise {
  guard !self.busy else { raise @frame.BatchFrameError::Busy }
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.remote.close()
  self.local.close()
  self.frame.close()
  self.state = Closed
}
