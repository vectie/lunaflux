///|
pub(all) enum PredictionPhase {
  Prime
  Predict
} derive(Eq, Debug)

///|
/// Two startup-prepared phase queues sharing explicit parent retirement.
/// No request data, model address or backend scheduling policy is inferred here.
pub struct PredictionPhaseExecution {
  priv prime : @device.OrderedKernelExecutor
  priv predict : @device.OrderedKernelExecutor
  priv prime_count : Int
  priv predict_count : Int
  priv mut active : PredictionPhase?
  priv mut closed : Bool
}

///|
pub fn PredictionPhaseExecution::prepare(
  context : @device.Context,
  stream : @device.Stream,
  prime : ArrayView[@device.OrderedKernelLaunch],
  predict : ArrayView[@device.OrderedKernelLaunch],
) -> Self raise {
  let p = context.create_ordered_kernel_executor(stream, prime)
  let q = context.create_ordered_kernel_executor(stream, predict) catch {
    e => {
      p.close()
      raise e
    }
  }
  {
    prime: p,
    predict: q,
    prime_count: prime.length(),
    predict_count: predict.length(),
    active: None,
    closed: false,
  }
}

///|
pub fn PredictionPhaseExecution::submit(
  self : Self,
  phase : PredictionPhase,
) -> Unit raise {
  guard !self.closed && self.active is None else {
    raise @device.DeviceError::Busy(OrderedExecutorEnqueue)
  }
  let (queue, count) = match phase {
    Prime => (self.prime, self.prime_count)
    Predict => (self.predict, self.predict_count)
  }
  // Keep the active marker on partial submission failure: abort/close must drain.
  self.active = Some(phase)
  for index in 0..<count {
    queue.enqueue(index)
  }
  queue.record_completion()
}

///|
pub fn PredictionPhaseExecution::poll(self : Self) -> Bool raise {
  guard !self.closed && self.active is Some(phase) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  let queue = match phase {
    Prime => self.prime
    Predict => self.predict
  }
  if !queue.poll_completion() {
    return false
  }
  queue.reset()
  self.active = None
  true
}

///|
pub fn PredictionPhaseExecution::abort(self : Self) -> Unit raise {
  if self.closed {
    return
  }
  if self.active is Some(phase) {
    let queue = match phase {
      Prime => self.prime
      Predict => self.predict
    }
    queue.abort()
    self.active = None
  }
}

///|
pub fn PredictionPhaseExecution::close(self : Self) -> Unit raise {
  if self.closed {
    return
  }
  self.abort()
  self.predict.close()
  self.prime.close()
  self.closed = true
}
