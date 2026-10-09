///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
pub struct RotaryKvFrame {
  priv source : @source.RotaryKvCudaSource
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub(all) struct RotaryKvOutputs {
  queries : @device.Allocation
  shared_kv : @device.Allocation
}

///|
pub fn RotaryKvFrame::new(source : @source.RotaryKvCudaSource) -> Self {
  { source, buffers: [], functions: [], launches: [], state: Fresh, }
}

///|
pub fn RotaryKvFrame::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
}

///|
pub fn RotaryKvFrame::workspace_bytes(self : Self) -> Int64 {
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
/// All inputs are borrowed. Outputs have exact startup-bounded allocation;
/// preparation performs no queue submission or host/device synchronization.
pub fn RotaryKvFrame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  counts~ : @device.Allocation,
  positions~ : @device.Allocation,
  queries~ : @device.Allocation,
  shared_kv~ : @device.Allocation,
  maximum_workspace_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh && self.workspace_bytes() <= maximum_workspace_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.state = Failed
  let (_, _, _, _, rows, _) = self.source.plan().geometry()
  let sizes = self.source.plan().frame_bytes()
  let count = region(counts, 20L, 4L)
  let position = region(positions, rows.to_int64() * 4L, 4L)
  let q = region(queries, sizes[0], 2L)
  let kv = region(shared_kv, sizes[1], 2L)
  for bytes in sizes {
    self.buffers.push(context.allocate(bytes))
  }
  let qo = region(self.buffers[0], sizes[0], 2L)
  let ko = region(self.buffers[1], sizes[1], 2L)
  let args = [[count, position, q, kv, qo, ko], [count, ko]]
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
        @device.KernelArguments::new(args[stage]),
      ),
    )
  }
  self.state = Prepared
}

///|
pub fn RotaryKvFrame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn RotaryKvFrame::outputs(self : Self) -> RotaryKvOutputs raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  { queries: self.buffers[0], shared_kv: self.buffers[1], }
}

///|
/// Close the borrowing ordered executor before releasing these effects.
pub fn RotaryKvFrame::close(self : Self) -> Unit raise {
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
