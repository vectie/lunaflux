///|
priv enum State {
  Fresh
  Prepared
  Failed
  Closed
}

///|
pub struct RotaryHadamardFp4Frame {
  priv source : @source.RotaryHadamardFp4CudaSource
  priv buffers : Array[@device.Allocation]
  priv functions : Array[@device.Function]
  priv launches : Array[@device.OrderedKernelLaunch]
  priv mut state : State
}

///|
pub fn RotaryHadamardFp4Frame::new(
  source : @source.RotaryHadamardFp4CudaSource,
) -> Self {
  { source, buffers: [], functions: [], launches: [], state: Fresh, }
}

///|
pub fn RotaryHadamardFp4Frame::workspace_bytes(self : Self) -> Int64 {
  self.source.plan().workspace_bytes()
}

///|
pub fn RotaryHadamardFp4Frame::source_bytes(self : Self) -> Bytes {
  self.source.source_bytes()
}

///|
fn region(
  allocation : @device.Allocation,
  bytes : Int64,
  alignment : Int64,
) -> @device.KernelArgument raise {
  @device.KernelArgument::device_region(
    allocation,
    offset=0L,
    byte_count=bytes,
    alignment~,
  )
}

///|
pub fn RotaryHadamardFp4Frame::prepare(
  self : Self,
  context : @device.Context,
  aot : @device.Module,
  counts~ : @device.Allocation,
  positions~ : @device.Allocation,
  input~ : @device.Allocation,
  maximum_workspace_bytes~ : Int64,
) -> Unit raise {
  guard self.state is Fresh && self.workspace_bytes() <= maximum_workspace_bytes else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.state = Failed
  let sizes = self.source.plan().frame_bytes()
  let (_, _, _, rows, _) = self.source.plan().rotary().geometry()
  let c = region(counts, 20L, 4L)
  let p = region(positions, rows.to_int64() * 4L, 4L)
  let x = region(input, sizes[0], 2L)
  for bytes in sizes {
    self.buffers.push(context.allocate(bytes))
  }
  let y = region(self.buffers[0], sizes[0], 2L)
  let z = region(self.buffers[1], sizes[1], 2L)
  let arguments = [[c, p, x, y], [c, y, z], [c, z]]
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
pub fn RotaryHadamardFp4Frame::launches(
  self : Self,
) -> ArrayView[@device.OrderedKernelLaunch] raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.launches[:]
}

///|
pub fn RotaryHadamardFp4Frame::output(self : Self) -> @device.Allocation raise {
  guard self.state is Prepared else {
    raise @device.DeviceError::InvalidArgument(OrderedExecutorCreate)
  }
  self.buffers[1]
}

///|
pub fn RotaryHadamardFp4Frame::close(self : Self) -> Unit raise {
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
