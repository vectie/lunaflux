///|
priv enum Phase {
  Empty
  Prepared
  Submitted
  Complete
  Failed
  Cancelled
  Closed
} derive(Eq)

///|
pub struct VisionRequest {
  priv source : VisionSource
  priv functions : Array[@device.Function]
  priv allocations : Array[@device.Allocation]
  priv mut weights : VisionWeights?
  priv mut queue : @device.OrderedKernelExecutor?
  priv mut phase : Phase
}

///|
pub fn VisionRequest::new(source : VisionSource) -> Self {
  {
    source,
    functions: [],
    allocations: [],
    weights: None,
    queue: None,
    phase: Empty,
  }
}

///|
/// One request owns its scratch and all four feature outputs. Partial failures
/// retain resources; close this owner before retrying with a new request.
pub fn VisionRequest::prepare(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  aot_module : @device.Module,
  weights : VisionWeights,
  inputs : ArrayView[@vision.ProcessedPixels],
  max_device_bytes~ : Int64,
) -> Unit raise {
  guard self.phase == Empty && weights.ready && weights.leases < 2147483647 else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let p = self.source.plan
  guard inputs.length() == p.grids.length() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let pixel_bytes = p.rows.to_int64() * 1536L * 2L
  guard p.workspace_bytes + p.output_bytes + pixel_bytes <= max_device_bytes else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  for i, input in inputs {
    let g = input.grid
    guard g == p.grids[i] &&
      input.patches.length().to_int64() ==
      g.time.to_int64() * g.height.to_int64() * g.width.to_int64() * 1536L * 2L else {
      raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
    }
  }
  self.phase = Failed
  weights.leases += 1
  self.weights = Some(weights)
  for symbol in self.source.symbols {
    self.functions.push(aot_module.load_function(@utf8.encode(symbol)))
  }
  fn allocate(bytes : Int64) -> @device.Allocation raise @device.DeviceError {
    let a = context.allocate(bytes)
    self.allocations.push(a)
    a
  }
  let workspace = allocate(p.workspace_bytes)
  let output = allocate(p.output_bytes)
  let patches = allocate(pixel_bytes)
  for input in inputs; offset = 0L {
    let bytes = input.patches.length().to_int64()
    patches.copy_from_host(input.patches, 0L, offset, bytes)
    continue offset + bytes
  } nobreak {
    ()
  }
  let launches = self.source.bind(
    self.functions,
    weights.allocations[0],
    weights.allocations[1:28],
    weights.allocations[28:32],
    patches~,
    workspace~,
    output~,
  )
  self.queue = Some(
    context.create_ordered_kernel_executor(
      stream,
      launches,
      policy=OrderedKernelEagerOnly,
    ),
  )
  self.phase = Prepared
}

///|
pub fn VisionRequest::submit(self : Self) -> Unit raise @device.DeviceError {
  guard self.phase == Prepared && self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  self.phase = Failed
  for i in 0..<queue.kernel_count() {
    queue.enqueue(i)
  }
  queue.record_completion()
  self.phase = Submitted
}

///|
pub fn VisionRequest::poll(self : Self) -> Bool raise @device.DeviceError {
  if self.phase == Complete {
    return true
  }
  guard self.phase == Submitted && self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  self.phase = Failed
  let done = queue.poll_completion()
  self.phase = if done { Complete } else { Submitted }
  done
}

///|
/// Borrow main+deepstack features after completion. Text consumer queues must
/// close before this request; device leases enforce that release ordering.
pub fn[T] VisionRequest::with_output(
  self : Self,
  consume : (@device.Allocation, @vision.VisionPlan) -> T raise @device.DeviceError,
) -> T raise @device.DeviceError {
  guard self.phase == Complete && self.allocations.length() == 3 else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  consume(self.allocations[1], self.source.plan)
}

///|
pub fn VisionRequest::cancel(self : Self) -> Unit raise @device.DeviceError {
  if self.phase == Closed || self.phase == Cancelled {
    return
  }
  self.phase = Failed
  if self.queue is Some(queue) {
    queue.abort()
  }
  self.phase = Cancelled
}

///|
pub fn VisionRequest::close(self : Self) -> Unit raise @device.DeviceError {
  if self.phase == Closed {
    return
  }
  self.phase = Failed
  if self.queue is Some(queue) {
    queue.abort()
    queue.close()
    self.queue = None
  }
  while !self.allocations.is_empty() {
    let i = self.allocations.length() - 1
    self.allocations[i].close()
    self.allocations.remove(i) |> ignore
  }
  while !self.functions.is_empty() {
    let i = self.functions.length() - 1
    self.functions[i].close()
    self.functions.remove(i) |> ignore
  }
  if self.weights is Some(weights) {
    weights.leases -= 1
    self.weights = None
  }
  self.phase = Closed
}
