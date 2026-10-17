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
  Graph
  Failed
  Closed
} derive(Eq)

///|
pub struct CollectiveKernelProgram {
  priv queues : ReadOnlyArray[@device.OrderedKernelExecutor]
  priv stream : @device.Stream
  priv owner : @device.TensorCollectiveOwner
  priv boundaries : ReadOnlyArray[ReductionBoundary]
  priv plan : @parallel.OrderedSubmissionPlan
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
  retirement? : @parallel.RetirementPolicy = EachCollective,
  replay? : @parallel.ReplayPolicy = Eager,
  replay_budget_bytes? : Int64 = 0L,
  capture_timeout_millis? : UInt64 = 60000UL,
) -> Self raise @device.DeviceError {
  guard !boundaries.is_empty() &&
    !launches.is_empty() &&
    maximum_rows > 0 &&
    capture_timeout_millis > 0UL else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  owner.validate_site_count(boundaries.length())
  let plan = @parallel.OrderedSubmissionPlan::new(
    kernel_count=launches.length(),
    boundaries.map(b => b.before_kernel),
    retirement~,
    replay~,
  ) catch {
    _ => raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
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
  if retirement == WholeFrame {
    owner.prepare_deferred()
  }
  let reserve = plan.replay_reserve_bytes(
    bytes_per_kernel=16384L,
    bytes_per_interval=65536L,
  ) catch {
    _ => raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  guard reserve <= replay_budget_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let queues : Array[@device.OrderedKernelExecutor] = []
  errdefer (for queue in queues.rev() {
    queue.abort() catch {
      _ => ()
    }
    queue.close() catch {
      _ => ()
    }
  })
  if replay == CaptureIntervals {
    for i in 0..<plan.interval_count() {
      let (first, end) = plan.interval(i)
      queues.push(
        context.create_ordered_kernel_executor(
          stream,
          launches[first:end],
          policy=OrderedKernelCaptureRequired,
        ),
      )
    }
  } else {
    queues.push(context.create_ordered_kernel_executor(stream, launches))
  }
  if replay == CapturePaddedCollectives {
    capture_collective_frame(
      queues[0],
      stream,
      boundaries,
      owner,
      generation,
      maximum_rows,
      launches.length(),
      capture_timeout_millis,
    )
  }
  {
    queues: ReadOnlyArray::from_array(queues),
    stream,
    owner,
    boundaries: ReadOnlyArray::from_array(boundaries),
    plan,
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
fn CollectiveKernelProgram::enqueue_interval(
  self : Self,
  interval : Int,
) -> Unit raise @device.DeviceError {
  let first = self.plan.interval_start(interval)
  let end = self.plan.interval_end(interval)
  guard self.next_kernel == first else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  if self.plan.replay_policy() == CaptureIntervals {
    let queue = self.queues[interval]
    queue.launch_captured()
    queue.record_completion()
    self.next_kernel = end
    return
  }
  while self.next_kernel < end {
    self.queues[0].enqueue(self.next_kernel)
    self.next_kernel += 1
  }
  if interval == self.plan.interval_count() - 1 {
    self.queues[0].record_completion()
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
  if self.plan.replay_policy() == CapturePaddedCollectives {
    self.owner.begin_graph(self.generation, plan_sequence, global_sequence)
    self.queues[0].launch_captured()
    self.queues[0].record_completion()
    self.state = Graph
    return
  }
  self.enqueue_interval(0)
  self.reduce()
}

///|
/// Issue consumers as soon as NCCL has submitted into the same stream. No
/// blocking GPU wait, launch-array construction, or readback enters progress.
pub fn CollectiveKernelProgram::poll(
  self : Self,
) -> Bool raise @device.DeviceError {
  if self.state == Graph {
    errdefer {
      self.state = Failed
    }
    if !self.queues[0].poll_completion() {
      return false
    }
    if !self.owner.complete_graph() {
      return false
    }
    self.queues[0].reset()
    self.state = Idle
    return true
  }
  guard self.state == Submission ||
    self.state == Completion ||
    self.state == Tail else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  errdefer {
    self.state = Failed
  }
  if self.state == Submission {
    let submitted = if self.plan.retirement_policy() == WholeFrame {
      self.owner.defer_submitted()
    } else {
      self.owner.poll_submitted()
    }
    if !submitted {
      return false
    }
    self.enqueue_interval(self.site + 1)
    if self.plan.retirement_policy() == WholeFrame {
      self.site += 1
      self.global_sequence += 1UL
      if self.site < self.boundaries.length() {
        self.reduce()
        return false
      }
      self.state = Tail
    } else {
      self.state = Completion
    }
  }
  if self.plan.retirement_policy() == WholeFrame && !self.owner.poll_deferred() {
    return false
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
  if self.queues[self.queues.length() - 1].poll_completion() {
    // The last event follows every same-stream graph and collective. No
    // segment is rearmed early, including on request cancellation.
    for i in 0..<self.queues.length() {
      if !self.queues[i].poll_completion() {
        return false
      }
    }
    for i in 0..<self.queues.length() {
      self.queues[i].reset()
    }
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
  for queue in self.queues.rev() {
    queue.close()
  }
  if self.plan.replay_policy() == CapturePaddedCollectives {
    self.owner.release_graph()
  }
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
  if self.plan.replay_policy() == CapturePaddedCollectives {
    for queue in self.queues.rev() {
      queue.discard_effect_graph()
    }
  }
  self.owner.abort()
  for queue in self.queues.rev() {
    queue.abort()
    queue.close()
  }
  self.state = Closed
}

///|
pub fn CollectiveKernelProgram::replay_graph_count(self : Self) -> Int {
  if self.plan.replay_policy() != Eager {
    self.queues.length()
  } else {
    0
  }
}
