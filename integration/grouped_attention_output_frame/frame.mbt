///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
/// Owns inverse-rotary, grouped-rank and hidden outputs. Compact weights,
/// positions, counts, attention result and the decoder queue are borrowed.
pub struct GroupedAttentionOutputFrame {
  priv source : @source.GroupedAttentionOutputCudaSource
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub fn GroupedAttentionOutputFrame::new(
  source : @source.GroupedAttentionOutputCudaSource,
) -> Self {
  { source, buffers: [], functions: [], launches: [], state: Fresh, }
}

///|
pub fn GroupedAttentionOutputFrame::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
}

///|
pub fn GroupedAttentionOutputFrame::workspace_bytes(self : Self) -> Int64 {
  self.source.plan().workspace_bytes()
}

///|
fn region(
  buffer : @device.Allocation,
  offset : Int64,
  bytes : Int64,
  alignment : Int64,
) -> @device.KernelArgument raise {
  @device.KernelArgument::device_region(
    buffer,
    offset~,
    byte_count=bytes,
    alignment~,
  )
}

///|
pub fn GroupedAttentionOutputFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  counts~ : @device.Allocation,
  positions~ : @device.Allocation,
  attention~ : @device.Allocation,
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
  let (_, _, _, rows) = plan.geometry()
  let count = region(counts, 0L, 20L, 4L)
  let position = region(positions, 0L, rows.to_int64() * 4L, 4L)
  let input = region(attention, 0L, plan.frame_bytes()[0], 2L)
  let matrices = plan.weight_representations()
  let w = [
    region(weights[0], 0L, matrices[0].payload_bytes(), 1L),
    region(
      weights[0],
      matrices[0].payload_bytes(),
      matrices[0].scale_bytes(),
      1L,
    ),
    region(weights[1], 0L, matrices[1].payload_bytes(), 1L),
    region(
      weights[1],
      matrices[1].payload_bytes(),
      matrices[1].scale_bytes(),
      1L,
    ),
  ]
  for bytes in plan.frame_bytes() {
    self.buffers.push(context.allocate(bytes))
  }
  let b = self.buffers.mapi((index, allocation) => {
    region(allocation, 0L, plan.frame_bytes()[index], 2L)
  })
  let arguments = [
    [count, position, input, b[0]],
    [count, b[0], w[0], w[1], b[1]],
    [count, b[1], w[2], w[3], b[2]],
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
  self.state = Prepared
}

///|
pub fn GroupedAttentionOutputFrame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn GroupedAttentionOutputFrame::output(
  self : Self,
) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[2]
}

///|
pub fn GroupedAttentionOutputFrame::close(self : Self) -> Unit raise {
  if self.state is Closed {
    return
  }
  self.state = Failed
  self.launches.clear()
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
