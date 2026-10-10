///|
pub(all) enum PredictionPhase {
  Prime
  Predict
  PredictCommitted
} derive(Eq, Debug)

///|
/// Startup-prepared phase queues sharing explicit parent retirement; the
/// committed-prefix queue is optional for consumers that only append one row.
/// No request data, model address or backend scheduling policy is inferred here.
pub struct PredictionPhaseExecution {
  priv prime : @device.OrderedKernelExecutor
  priv predict : @device.OrderedKernelExecutor
  priv committed : @device.OrderedKernelExecutor?
  priv prime_count : Int
  priv predict_count : Int
  priv committed_count : Int
  priv mut active : PredictionPhase?
  priv mut closed : Bool
}

///|
pub fn PredictionPhaseExecution::prepare(
  context : @device.Context,
  stream : @device.Stream,
  prime : ArrayView[@device.OrderedKernelLaunch],
  predict : ArrayView[@device.OrderedKernelLaunch],
  committed? : ArrayView[@device.OrderedKernelLaunch],
) -> Self raise {
  let p = context.create_ordered_kernel_executor(stream, prime)
  let q = context.create_ordered_kernel_executor(stream, predict) catch {
    e => {
      p.close()
      raise e
    }
  }
  let r = match committed {
    None => None
    Some(table) =>
      Some(
        context.create_ordered_kernel_executor(stream, table) catch {
          e => {
            q.close()
            p.close()
            raise e
          }
        },
      )
  }
  {
    prime: p,
    predict: q,
    committed: r,
    prime_count: prime.length(),
    predict_count: predict.length(),
    committed_count: match committed {
      Some(table) => table.length()
      None => 0
    },
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
    PredictCommitted =>
      match self.committed {
        Some(queue) => (queue, self.committed_count)
        None =>
          raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
      }
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
    PredictCommitted =>
      match self.committed {
        Some(queue) => queue
        None => raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
      }
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
      PredictCommitted =>
        match self.committed {
          Some(queue) => queue
          None =>
            raise @device.DeviceError::InvalidArgument(OrderedExecutorAbort)
        }
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
  if self.committed is Some(queue) {
    queue.close()
  }
  self.predict.close()
  self.prime.close()
  self.closed = true
}
