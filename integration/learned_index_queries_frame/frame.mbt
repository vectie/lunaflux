///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
pub struct LearnedIndexQueriesFrame {
  priv source : @source.LearnedIndexQueriesCudaSource
  priv transform : @transform.RotaryHadamardFp4Frame
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub(all) struct LearnedIndexQueryOutputs {
  queries : @device.Allocation
  head_weights : @device.Allocation
}

///|
pub fn LearnedIndexQueriesFrame::new(
  source : @source.LearnedIndexQueriesCudaSource,
) -> Self {
  {
    source,
    transform: @transform.RotaryHadamardFp4Frame::new(source.transform()),
    buffers: [],
    functions: [],
    launches: [],
    state: Fresh,
  }
}

///|
pub fn LearnedIndexQueriesFrame::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
}

///|
pub fn LearnedIndexQueriesFrame::workspace_bytes(self : Self) -> Int64 {
  self.source.plan().workspace_bytes()
}

///|
fn region(
  buffer : @device.Allocation,
  bytes : Int64,
  alignment : Int64,
) -> @device.KernelArgument raise {
  @device.KernelArgument::device_region(
    buffer,
    offset=0L,
    byte_count=bytes,
    alignment~,
  )
}

///|
pub fn LearnedIndexQueriesFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  counts~ : @device.Allocation,
  positions~ : @device.Allocation,
  hidden~ : @device.Allocation,
  low_rank_queries~ : @device.Allocation,
  weights~ : ArrayView[@device.Allocation],
  maximum_workspace_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh &&
    weights.length() == 2 &&
    self.workspace_bytes() <= maximum_workspace_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.state = Failed
  let plan = self.source.plan()
  let (hidden_width, rank, _, _, rows) = plan.geometry()
  let count = region(counts, 20L, 4L)
  ignore(region(positions, rows.to_int64() * 4L, 4L))
  let hidden_input = region(
    hidden,
    rows.to_int64() * hidden_width.to_int64() * 2L,
    2L,
  )
  let qr = region(low_rank_queries, rows.to_int64() * rank.to_int64() * 2L, 2L)
  let sizes = plan.weight_bytes()
  let query_weight = region(weights[0], sizes[0], 1L)
  let query_scale = @device.KernelArgument::device_region(
    weights[0],
    offset=sizes[0],
    byte_count=sizes[1],
    alignment=1L,
  )
  let head_weight = region(weights[1], sizes[2], 2L)
  for size in plan.frame_bytes() {
    self.buffers.push(context.allocate(size))
  }
  let outputs = self.buffers.mapi((index, buffer) => {
    region(buffer, plan.frame_bytes()[index], 2L)
  })
  let arguments = [
    [count, qr, query_weight, query_scale, outputs[0]],
    [count, hidden_input, head_weight, outputs[1]],
    [count, outputs[1], outputs[2]],
  ]
  for stage in 0..<3 {
    let function = aot.load_function(@utf8.encode(self.source.symbol(stage)))
    self.functions.push(function)
    let (x, y, threads) = self.source.launch(stage)
    self.launches.push(
      @device.OrderedKernelLaunch::new(
        function,
        @device.KernelLaunchDimensions::new(
          grid_x=x,
          grid_y=y,
          grid_z=1,
          block_x=threads,
          block_y=1,
          block_z=1,
          shared_memory_bytes=0,
        ),
        @device.KernelArguments::new(arguments[stage]),
      ),
    )
  }
  self.transform.prepare(
    context,
    aot,
    counts~,
    positions~,
    input=self.buffers[0],
    maximum_workspace_bytes=plan.transform().workspace_bytes(),
  )
  self.launches.append(self.transform.launches())
  self.state = Prepared
}

///|
pub fn LearnedIndexQueriesFrame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn LearnedIndexQueriesFrame::outputs(
  self : Self,
) -> LearnedIndexQueryOutputs raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  { queries: self.transform.output(), head_weights: self.buffers[2], }
}

///|
pub fn LearnedIndexQueriesFrame::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.launches.clear()
  self.transform.close()
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
