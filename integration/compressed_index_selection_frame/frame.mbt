///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
pub struct CompressedIndexSelectionFrame {
  priv source : @source.CompressedIndexSelectionCudaSource
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub(all) struct CompressedIndexSelectionOutputs {
  row_ids : @device.Allocation
  row_counts : @device.Allocation
}

///|
pub fn CompressedIndexSelectionFrame::new(
  source : @source.CompressedIndexSelectionCudaSource,
) -> Self {
  { source, buffers: [], functions: [], launches: [], state: Fresh, }
}

///|
pub fn CompressedIndexSelectionFrame::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
}

///|
pub fn CompressedIndexSelectionFrame::workspace_bytes(self : Self) -> Int64 {
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
pub fn CompressedIndexSelectionFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  counts~ : @device.Allocation,
  positions~ : @device.Allocation,
  cache_counts~ : @device.Allocation,
  cache_append~ : @device.Allocation,
  queries~ : @device.Allocation,
  head_weights~ : @device.Allocation,
  cache_values~ : @device.Allocation,
  offsets~ : @device.Allocation,
  maximum_workspace_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh && self.workspace_bytes() <= maximum_workspace_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.state = Failed
  let plan = self.source.plan()
  let (heads, width, _, rows, _, _) = plan.geometry()
  let c = region(counts, 20L, 4L)
  let p = region(positions, rows.to_int64() * 4L, 4L)
  let cc = region(cache_counts, 20L, 4L)
  let ca = region(cache_append, 12L, 4L)
  let q = region(
    queries,
    rows.to_int64() * heads.to_int64() * width.to_int64() * 2L,
    2L,
  )
  let w = region(head_weights, rows.to_int64() * heads.to_int64() * 2L, 2L)
  let k = region(
    cache_values,
    Int::max(plan.cache_capacity(), 1).to_int64() * width.to_int64() * 2L,
    2L,
  )
  let o = region(offsets, rows.to_int64() * 4L, 4L)
  for bytes in plan.frame_bytes() {
    self.buffers.push(context.allocate(bytes))
  }
  let b = self.buffers.mapi((index, buffer) => {
    region(buffer, plan.frame_bytes()[index], if index == 0 { 2L } else { 4L })
  })
  let arguments = [
    [c, p, cc, ca, q, w, k, b[0]],
    [c, p, cc, ca, o, b[0], b[2], b[1]],
  ]
  for stage in 0..<2 {
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
pub fn CompressedIndexSelectionFrame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn CompressedIndexSelectionFrame::outputs(
  self : Self,
) -> CompressedIndexSelectionOutputs raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  { row_ids: self.buffers[1], row_counts: self.buffers[2], }
}

///|
pub fn CompressedIndexSelectionFrame::close(self : Self) -> Unit raise {
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
