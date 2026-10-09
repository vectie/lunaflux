///|
pub fn GlmDecoderSlice::submit(
  self : Self,
  live_rows~ : Int,
) -> Unit raise @device.DeviceError {
  guard self.status is Ready && self.queue is Some(queue) else {
    raise @device.DeviceError::Busy(OrderedExecutorEnqueue)
  }
  guard live_rows > 0 && live_rows <= self.maximum_rows else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.status = Failed
  for stage in 0..<self.launch_count {
    queue.enqueue(stage)
  }
  queue.record_completion()
  self.status = Active
}

///|
pub fn GlmDecoderSlice::poll(self : Self) -> Bool raise @device.DeviceError {
  guard self.status is Active && self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  self.status = Failed
  if queue.poll_completion() {
    queue.reset()
    self.status = Ready
    true
  } else {
    self.status = Active
    false
  }
}

///|
pub fn GlmDecoderSlice::close(self : Self) -> Unit raise @device.DeviceError {
  if self.status is Closed {
    return
  }
  self.status = Failed
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
