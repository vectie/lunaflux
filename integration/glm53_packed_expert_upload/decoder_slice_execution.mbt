///|
pub fn GlmDecoderSlice::submit(
  self : Self,
  live_rows~ : Int,
) -> Unit raise @device.DeviceError {
  guard self.status is Ready else {
    raise @device.DeviceError::Busy(OrderedExecutorEnqueue)
  }
  guard live_rows > 0 && live_rows <= self.maximum_rows else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.status = Failed
  if self.collective_queue is Some(queue) {
    guard self.plan_sequence < 0xffffffffffffffffUL &&
      self.global_sequence <=
      0xffffffffffffffffUL - self.collective_count.to_uint64() else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
    }
    queue.submit(
      live_rows~,
      plan_sequence=self.plan_sequence + 1UL,
      global_sequence=self.global_sequence,
    )
  } else if self.queue is Some(queue) {
    for stage in 0..<self.launch_count {
      queue.enqueue(stage)
    }
    queue.record_completion()
  } else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.status = Active
}

///|
pub fn GlmDecoderSlice::poll(self : Self) -> Bool raise @device.DeviceError {
  guard self.status is Active else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  self.status = Failed
  let complete = if self.collective_queue is Some(queue) {
    queue.poll()
  } else if self.queue is Some(queue) {
    if queue.poll_completion() {
      queue.reset()
      true
    } else {
      false
    }
  } else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  if complete {
    if self.collective_queue is Some(_) {
      self.plan_sequence += 1UL
      self.global_sequence += self.collective_count.to_uint64()
    }
    self.status = Ready
    true
  } else {
    self.status = Active
    false
  }
}

///|
fn GlmDecoderSlice::abort_execution(
  self : Self,
) -> Unit raise @device.DeviceError {
  if self.status is Closed {
    return
  }
  self.status = Failed
  if self.collective_queue is Some(queue) {
    // The program owns capture references and the communicator ordering:
    // discard graphs, abort communication, then drain the retained streams.
    queue.abort()
    self.collective_queue = None
  }
  if self.queue is Some(queue) {
    queue.abort()
    queue.close()
    self.queue = None
  }
}

///|
pub fn GlmDecoderSlice::close(self : Self) -> Unit raise @device.DeviceError {
  if self.status is Closed {
    return
  }
  let idle = self.status is Ready
  self.status = Failed
  if self.collective_queue is Some(queue) {
    if idle {
      queue.close()
    } else {
      queue.abort()
    }
    self.collective_queue = None
  }
  if self.queue is Some(queue) {
    queue.abort()
    queue.close()
    self.queue = None
  }
  for layer in self.layers.rev() {
    layer.close()
  }
  while self.buffers.last() is Some(buffer) {
    buffer.close()
    ignore(self.buffers.pop())
  }
  self.status = Closed
}
