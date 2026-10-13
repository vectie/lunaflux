///|
/// A startup-bound sum inserted before one launch. Kernels, buffers, stream,
/// and communicator are borrowed and must outlive the program.
pub struct ReductionBoundary {
  priv before_kernel : Int
  priv width : Int
  priv operation_id : Int
  priv buffer : @device.Allocation
}

///|
pub fn ReductionBoundary::new(
  before_kernel~ : Int,
  width~ : Int,
  operation_id~ : Int,
  buffer~ : @device.Allocation,
) -> Self {
  { before_kernel, width, operation_id, buffer, }
}

///|
priv enum State {
  Idle
  Submission
  Completion
  Tail
  Failed
  Closed
} derive(Eq)

///|
pub struct CollectiveKernelProgram {
  priv queue : @device.OrderedKernelExecutor
  priv stream : @device.Stream
  priv owner : @device.TensorCollectiveOwner
  priv boundaries : ReadOnlyArray[ReductionBoundary]
  priv generation : UInt64
  priv maximum_rows : Int
  priv mut state : State
  priv mut site : Int
  priv mut next_kernel : Int
  priv mut rows : Int
  priv mut plan_sequence : UInt64
  priv mut global_sequence : UInt64
}

///|
pub fn CollectiveKernelProgram::new(
  context : @device.Context,
  stream : @device.Stream,
  launches : ArrayView[@device.OrderedKernelLaunch],
  boundaries : ArrayView[ReductionBoundary],
  owner~ : @device.TensorCollectiveOwner,
  generation~ : UInt64,
  maximum_rows~ : Int,
) -> Self raise @device.DeviceError {
  guard !boundaries.is_empty() && !launches.is_empty() && maximum_rows > 0 else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  owner.validate_site_count(boundaries.length())
  for i, boundary in boundaries {
    guard boundary.before_kernel > 0 &&
      boundary.before_kernel < launches.length() &&
      (i == 0 || boundary.before_kernel > boundaries[i - 1].before_kernel) else {
      raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
    }
    owner.validate_sum_site(
      site_sequence=i + 1,
      operation_id=boundary.operation_id,
      maximum_rows~,
      width=boundary.width,
    )
    context.validate_allocation_region(
      boundary.buffer,
      offset=0L,
      byte_count=maximum_rows.to_int64() * boundary.width.to_int64() * 4L,
      alignment=4L,
    )
  }
  {
    queue: context.create_ordered_kernel_executor(stream, launches),
    stream,
    owner,
    boundaries: ReadOnlyArray::from_array(boundaries),
    generation,
    maximum_rows,
    state: Idle,
    site: 0,
    next_kernel: 0,
    rows: 0,
    plan_sequence: 0UL,
    global_sequence: 0UL,
  }
}

///|
fn CollectiveKernelProgram::enqueue_until(
  self : Self,
  end : Int,
) -> Unit raise @device.DeviceError {
  while self.next_kernel < end {
    self.queue.enqueue(self.next_kernel)
    self.next_kernel += 1
  }
}

///|
fn CollectiveKernelProgram::reduce(
  self : Self,
) -> Unit raise @device.DeviceError {
  let boundary = self.boundaries[self.site]
  self.owner.submit(
    generation=self.generation,
    plan_sequence=self.plan_sequence,
    global_sequence=self.global_sequence,
    site_sequence=self.site + 1,
    operation_id=boundary.operation_id,
    kind=F32SumAllReduce,
    live_query_tokens=self.rows,
    send=boundary.buffer,
    send_offset=0L,
    receive=boundary.buffer,
    receive_offset=0L,
    stream=self.stream,
  )
  self.state = Submission
}

///|
pub fn CollectiveKernelProgram::submit(
  self : Self,
  live_rows~ : Int,
  plan_sequence~ : UInt64,
  global_sequence~ : UInt64,
) -> Unit raise @device.DeviceError {
  guard self.state == Idle && live_rows > 0 && live_rows <= self.maximum_rows else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.state = Failed
  self.rows = live_rows
  self.plan_sequence = plan_sequence
  self.global_sequence = global_sequence
  self.site = 0
  self.next_kernel = 0
  self.enqueue_until(self.boundaries[0].before_kernel)
  self.reduce()
}

///|
/// Issue consumers as soon as NCCL has submitted into the same stream. No
/// blocking GPU wait, launch-array construction, or readback enters progress.
pub fn CollectiveKernelProgram::poll(
  self : Self,
) -> Bool raise @device.DeviceError {
  guard self.state == Submission ||
    self.state == Completion ||
    self.state == Tail else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  errdefer {
    self.state = Failed
  }
  if self.state == Submission {
    if !self.owner.poll_submitted() {
      return false
    }
    let end = if self.site + 1 < self.boundaries.length() {
      self.boundaries[self.site + 1].before_kernel
    } else {
      self.queue.kernel_count()
    }
    self.enqueue_until(end)
    if self.site + 1 == self.boundaries.length() {
      self.queue.record_completion()
    }
    self.state = Completion
  }
  if self.state == Completion {
    if !self.owner.poll() {
      return false
    }
    self.site += 1
    self.global_sequence += 1UL
    if self.site < self.boundaries.length() {
      self.reduce()
      return false
    }
    self.state = Tail
  }
  if self.queue.poll_completion() {
    self.queue.reset()
    self.state = Idle
    true
  } else {
    false
  }
}

///|
pub fn CollectiveKernelProgram::close(
  self : Self,
) -> Unit raise @device.DeviceError {
  if self.state == Closed {
    return
  }
  guard self.state == Idle else {
    raise @device.DeviceError::Busy(OrderedExecutorClose)
  }
  self.queue.close()
  self.state = Closed
}

///|
pub fn CollectiveKernelProgram::abort(
  self : Self,
) -> Unit raise @device.DeviceError {
  if self.state == Closed {
    return
  }
  self.state = Failed
  self.owner.abort()
  self.queue.abort()
  self.queue.close()
  self.state = Closed
}
