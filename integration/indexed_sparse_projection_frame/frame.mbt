///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
/// Owns intermediates/functions, borrows the hidden frame and twelve weight
/// allocations. All launch argument binding happens at startup, not token-step.
pub struct IndexedSparseProjectionFrame {
  priv source : @source.IndexedSparseProjectionCudaSource
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut counts : @device.Allocation?
  priv mut state : State
}

///|
/// Borrowed device ports, owned by the projection frame, never by this view.
pub(all) struct SparseProjectionOutputs {
  queries : @device.Allocation
  keys : @device.Allocation
  values : @device.Allocation
  index_queries : @device.Allocation
  index_keys : @device.Allocation
  index_head_weights : @device.Allocation
  pool_gate_scores : @device.Allocation
}

///|
pub fn IndexedSparseProjectionFrame::new(
  source : @source.IndexedSparseProjectionCudaSource,
) -> Self {
  {
    source,
    buffers: [],
    functions: [],
    launches: [],
    counts: None,
    state: Fresh,
  }
}

///|
pub fn IndexedSparseProjectionFrame::workspace_bytes(self : Self) -> Int64 {
  self.source.plan().workspace_bytes()
}

///|
fn region(
  buffer : @device.Allocation,
  bytes : Int64,
) -> @device.KernelArgument raise @device.DeviceError {
  @device.KernelArgument::device_region(
    buffer,
    offset=0L,
    byte_count=bytes,
    alignment=2L,
  )
}

///|
/// Weights follow the pure plan's weight_bytes order. They remain caller-owned.
pub fn IndexedSparseProjectionFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  maximum_workspace_bytes~ : Int64,
  query_counts~ : @device.Allocation,
  hidden~ : @device.Allocation,
  weights~ : ArrayView[@device.Allocation],
) -> Unit raise {
  guard self.state is Fresh &&
    weights.length() == 12 &&
    self.workspace_bytes() <= maximum_workspace_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.state = Failed
  let plan = self.source.plan()
  let sizes = plan.frame_bytes()
  let w_sizes = plan.weight_bytes()
  let (width, _, _) = plan.geometry()
  let (rows, _) = plan.attention().capacity()
  let counts = @device.KernelArgument::device_region(
    query_counts,
    offset=0L,
    byte_count=20L,
    alignment=4L,
  )
  let input = region(hidden, rows.to_int64() * width.to_int64() * 2L)
  let w = weights.mapi((i, buffer) => region(buffer, w_sizes[i]))
  for size in sizes {
    self.buffers.push(context.allocate(size))
  }
  let b = self.buffers.mapi((i, buffer) => region(buffer, sizes[i]))
  let args = [
    [counts, input, w[0], b[0]],
    [counts, b[0], w[1], b[1]],
    [counts, b[1], w[2], b[2]],
    [counts, input, w[3], b[3]],
    [counts, b[3], w[4], b[4]],
    [counts, b[4], w[5], b[5]],
    [counts, b[5], b[6], b[7]],
    [counts, b[1], w[6], b[8]],
    [counts, input, w[7], b[9]],
    [counts, b[9], w[8], w[9], b[10]],
    [counts, input, w[10], b[11]],
    [counts, input, w[11], b[12]],
  ]
  for stage in 0..<12 {
    let function = aot.load_function(@utf8.encode(self.source.symbol(stage)))
    self.functions.push(function)
    let (grid, block) = self.source.launch(stage)
    self.launches.push(
      @device.OrderedKernelLaunch::new(
        function,
        @device.KernelLaunchDimensions::new(
          grid_x=grid,
          grid_y=1,
          grid_z=1,
          block_x=block,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        @device.KernelArguments::new(args[stage]),
      ),
    )
  }
  self.counts = Some(query_counts)
  self.state = Prepared
}

///|
pub fn IndexedSparseProjectionFrame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise @device.DeviceError {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
/// Borrowed outputs for the history/attention composition; callers never close
/// them. Closing order is queue, attention, history, then projection frame.
pub fn IndexedSparseProjectionFrame::projected_outputs(
  self : Self,
) -> SparseProjectionOutputs raise @device.DeviceError {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  let b = self.buffers
  {
    queries: b[2],
    keys: b[6],
    values: b[7],
    index_queries: b[8],
    index_keys: b[10],
    index_head_weights: b[11],
    pool_gate_scores: b[12],
  }
}

///|
pub fn IndexedSparseProjectionFrame::close(
  self : Self,
) -> Unit raise @device.DeviceError {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.launches.clear()
  self.counts = None
  while self.functions.last() is Some(function) {
    function.close()
    ignore(self.functions.pop())
  }
  while self.buffers.last() is Some(buffer) {
    buffer.close()
    ignore(self.buffers.pop())
  }
  self.state = Closed
}
