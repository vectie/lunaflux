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
/// One-shot request owner, with retryable release after any preparation failure.
pub struct EncoderRequest {
  priv source : TextEncoderSource
  priv config : @text_plan.EncoderConfig
  priv functions : Array[@device.Function]
  priv allocations : Array[@device.Allocation]
  priv mut weights : EncoderWeights?
  priv mut queue : @device.OrderedKernelExecutor?
  priv observations : Array[(@device.OrderedKernelExecutor, Int)]
  priv mut phase : Phase
}

///|
pub fn EncoderRequest::new(
  source : TextEncoderSource,
  config : @text_plan.EncoderConfig,
) -> Self raise @device.DeviceError {
  guard source.plan.epsilon == config.epsilon &&
    source.plan.rows <= config.max_positions else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  {
    source,
    config,
    functions: [],
    allocations: [],
    weights: None,
    queue: None,
    observations: [],
    phase: Empty,
  }
}

///|
fn EncoderRequest::prepare_execution(
  self : Self,
  context : @device.Context,
  stream : @device.Stream,
  aot_module : @device.Module,
  weights : EncoderWeights,
  ids : ArrayView[Int],
  max_device_bytes~ : Int64,
  max_table_bytes~ : Int64,
  observe_layers~ : Bool,
  visual? : (@device.Allocation, @text_plan.VisualPlacement),
) -> Unit raise {
  guard self.phase == Empty && weights.ready && weights.leases < 2147483647 else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let p = match visual {
    None => self.config.plan(ids)
    Some((_, placement)) => {
      self.config.visual_placement(
        ids,
        placement.token_positions[:],
        placement.rotary_positions[:],
      )
      |> ignore
      @text_plan.TextEncoderPlan::new(
        rows=ids.length(),
        epsilon=self.config.epsilon,
      )
    }
  }
  guard p.rows == self.source.plan.rows else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  let regions = p.workspace_regions()
  let end = regions[regions.length() - 1]
  let workspace_bytes = end.0 + end.1
  let positions = match visual {
    None => @text_plan.text_positions(p.rows)
    Some((_, placement)) => placement.rotary_positions
  }
  let visual_rows = match visual {
    None => 0
    Some((_, placement)) => placement.token_positions.length()
  }
  let tables = self.config.rotary_tables(
    positions[:],
    max_bytes=max_table_bytes,
  )
  let total = workspace_bytes +
    16L +
    p.rows.to_int64() * 4L +
    tables.cosine.length().to_int64() +
    tables.sine.length().to_int64() +
    visual_rows.to_int64() * 4L
  guard max_device_bytes > 0L && total <= max_device_bytes else {
    raise @device.DeviceError::InvalidArgument(AllocationCreate)
  }
  self.phase = Failed
  weights.leases += 1
  self.weights = Some(weights)
  for name in self.source.symbols {
    self.functions.push(aot_module.load_function(@utf8.encode(name)))
  }
  let workspace = context.allocate(workspace_bytes)
  self.allocations.push(workspace)
  fn upload(bytes : Bytes) -> @device.Allocation raise @device.DeviceError {
    let a = context.allocate(bytes.length().to_int64())
    self.allocations.push(a)
    a.copy_from_host(bytes, 0L, 0L, bytes.length().to_int64())
    a
  }
  let counts = upload(@text_plan.encode_i32([visual_rows, 0, 0, p.rows]))
  let token_ids = upload(@text_plan.encode_i32(ids))
  let cosine = upload(tables.cosine)
  let sine = upload(tables.sine)
  let layers : Array[(@text_plan.LayerWeights, @device.Allocation)] = []
  for layer in 0..<50 {
    layers.push(
      (@text_plan.layer_weights(layer), weights.allocations[layer + 1]),
    )
  }
  let visual_binding = match visual {
    None => None
    Some((features, placement)) =>
      Some(
        (
          visual_rows,
          upload(@text_plan.encode_i32(placement.token_positions[:])),
          features,
        ),
      )
  }
  let launches = self.source.bind(
    self.functions,
    layers,
    counts~,
    token_ids~,
    embedding=weights.allocations[0],
    workspace~,
    cosine~,
    sine~,
    visual?=visual_binding,
  )
  if observe_layers {
    let boundaries = hidden_state_boundaries(visual is Some(_))
    for (hidden, end) in boundaries; start = 0 {
      let queue = context.create_ordered_kernel_executor(
        stream,
        launches[start:end],
        policy=OrderedKernelEagerOnly,
      )
      self.observations.push((queue, hidden))
      continue end
    }
  } else {
    self.queue = Some(
      context.create_ordered_kernel_executor(
        stream,
        launches,
        policy=OrderedKernelEagerOnly,
      ),
    )
  }
  self.phase = Prepared
}

///|
pub fn EncoderRequest::submit(self : Self) -> Unit raise @device.DeviceError {
  guard self.phase == Prepared && self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorEnqueue)
  }
  let phase = self.phase
  self.phase = Failed
  self.phase = submit_once(phase, () => {
    for i in 0..<queue.kernel_count() {
      queue.enqueue(i)
    }
    queue.record_completion()
  })
}

///|
pub fn EncoderRequest::poll(self : Self) -> Bool raise @device.DeviceError {
  if self.phase == Complete {
    return true
  }
  guard self.phase == Submitted && self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorPoll)
  }
  let previous = self.phase
  self.phase = Failed
  let (phase, done) = poll_once(previous, () => queue.poll_completion())
  self.phase = phase
  done
}

///|
pub fn EncoderRequest::wait(self : Self) -> Unit raise @device.DeviceError {
  if self.phase == Complete {
    return
  }
  guard self.phase == Submitted && self.queue is Some(queue) else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorWait)
  }
  self.phase = Failed
  queue.wait_completion()
  self.phase = Complete
}

///|
pub fn EncoderRequest::cancel(self : Self) -> Unit raise @device.DeviceError {
  if self.phase == Closed || self.phase == Cancelled {
    return
  }
  self.phase = Failed
  if self.queue is Some(queue) {
    queue.abort()
  }
  for (queue, _) in self.observations {
    queue.abort()
  }
  self.phase = Cancelled
}

///|
pub fn[T] EncoderRequest::with_output(
  self : Self,
  consume : (@device.Allocation, Int64) -> T raise @device.DeviceError,
) -> T raise @device.DeviceError {
  guard self.phase == Complete && !self.allocations.is_empty() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  consume(self.allocations[0], 0L)
}

///|
/// Preparation-only destination borrowing for a downstream ordered queue.
/// Contents are NOT ready. Execute this encoder to completion before that queue.
pub fn[T] EncoderRequest::with_destination(
  self : Self,
  consume : (@device.Allocation, Int64) -> T raise @device.DeviceError,
) -> T raise @device.DeviceError {
  guard self.phase == Prepared && !self.allocations.is_empty() else {
    raise @device.DeviceError::InvalidArgument(AotKernelLaunch)
  }
  consume(self.allocations[0], 0L)
}

///|
/// Close downstream refiner queues before closing the encoder request.
pub fn EncoderRequest::close(self : Self) -> Unit raise @device.DeviceError {
  if self.phase == Closed {
    return
  }
  self.phase = Failed
  if self.queue is Some(queue) {
    queue.abort()
    queue.close()
    self.queue = None
  }
  while !self.observations.is_empty() {
    let last = self.observations.length() - 1
    let (queue, _) = self.observations[last]
    queue.abort()
    queue.close()
    self.observations.remove(last) |> ignore
  }
  while !self.allocations.is_empty() {
    let last = self.allocations.length() - 1
    self.allocations[last].close()
    self.allocations.remove(last) |> ignore
  }
  while !self.functions.is_empty() {
    let last = self.functions.length() - 1
    self.functions[last].close()
    self.functions.remove(last) |> ignore
  }
  if self.weights is Some(weights) {
    weights.leases -= 1
    self.weights = None
  }
  self.phase = Closed
}
