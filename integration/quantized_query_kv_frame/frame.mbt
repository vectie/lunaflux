///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
/// Owns six BF16 intermediates and six functions, not counts/hidden/weights or
/// the decoder queue. Outputs precede rotary/cache operations, not attention.
pub struct QuantizedQueryKvFrame {
  priv source : @source.QuantizedQueryKvCudaSource
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub(all) struct QueryKvProjectionOutputs {
  low_rank_queries : @device.Allocation
  queries : @device.Allocation
  shared_kv : @device.Allocation
}

///|
pub fn QuantizedQueryKvFrame::new(
  source : @source.QuantizedQueryKvCudaSource,
) -> Self {
  { source, buffers: [], functions: [], launches: [], state: Fresh, }
}

///|
pub fn QuantizedQueryKvFrame::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
}

///|
pub fn QuantizedQueryKvFrame::workspace_bytes(self : Self) -> Int64 {
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
/// Five compact weights: Query-A, query norm, Query-B, KV, KV norm. Each
/// quantized matrix retains its payload followed by its scale plane. Every
/// borrowed span is checked before scratch allocation; no matrix expansion.
pub fn QuantizedQueryKvFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  counts~ : @device.Allocation,
  hidden~ : @device.Allocation,
  weights~ : ArrayView[@device.Allocation],
  maximum_workspace_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh &&
    weights.length() == 5 &&
    self.workspace_bytes() <= maximum_workspace_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.state = Failed
  let plan = self.source.plan()
  let (width, _, _, _, rows) = plan.geometry()
  let count = region(counts, 20L, 4L)
  let input = region(hidden, rows.to_int64() * width.to_int64() * 2L, 2L)
  let sizes = plan.weight_bytes()
  let plane = (bank, offset, bytes) => {
    @device.KernelArgument::device_region(
      weights[bank],
      offset~,
      byte_count=bytes,
      alignment=1L,
    )
  }
  let w = [
    plane(0, 0L, sizes[0]),
    plane(0, sizes[0], sizes[1]),
    region(weights[1], sizes[2], 2L),
    plane(2, 0L, sizes[3]),
    plane(2, sizes[3], sizes[4]),
    plane(3, 0L, sizes[5]),
    plane(3, sizes[5], sizes[6]),
    region(weights[4], sizes[7], 2L),
  ]
  for size in plan.frame_bytes() {
    self.buffers.push(context.allocate(size))
  }
  let b = self.buffers.mapi((i, buffer) => {
    region(buffer, plan.frame_bytes()[i], 2L)
  })
  let args = [
    [count, input, w[0], w[1], b[0]],
    [count, b[0], w[2], b[1]],
    [count, b[1], w[3], w[4], b[2]],
    [count, b[2], b[3]],
    [count, input, w[5], w[6], b[4]],
    [count, b[4], w[7], b[5]],
  ]
  for stage in 0..<6 {
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
        @device.KernelArguments::new(args[stage]),
      ),
    )
  }
  self.state = Prepared
}

///|
pub fn QuantizedQueryKvFrame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn QuantizedQueryKvFrame::outputs(
  self : Self,
) -> QueryKvProjectionOutputs raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  {
    low_rank_queries: self.buffers[1],
    queries: self.buffers[3],
    shared_kv: self.buffers[5],
  }
}

///|
/// Close the borrowing decoder queue first, then this frame, then its weights.
pub fn QuantizedQueryKvFrame::close(self : Self) -> Unit raise {
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
